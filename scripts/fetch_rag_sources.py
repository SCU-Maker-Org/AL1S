#!/usr/bin/env python3
"""Fetch an allowlisted set of official technical documentation for RAG."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import ipaddress
import json
import os
import re
import socket
import sys
import tempfile
import tomllib
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import aiohttp
import yaml
from aiohttp.abc import AbstractResolver
from bs4 import BeautifulSoup, NavigableString, Tag

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "knowledge" / "sources.toml"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "rag_sources"

BUILTIN_OFFICIAL_HOSTS = frozenset(
    {
        "www.postgresql.org",
        "www.kernel.org",
        "kubernetes.io",
        "www.llvm.org",
        "docs.pytorch.org",
        "docs.nvidia.com",
        "docs.vllm.ai",
        "cocoonstack.github.io",
    }
)
ALLOWED_DOMAINS = frozenset(
    {
        "sys",
        "hpc",
        "compile",
        "distributed",
        "db",
        "storage",
        "ai_workload",
        "cloud",
        "security",
    }
)
ALLOWED_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml"})
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
SOURCE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")
CHARSET_PATTERN = re.compile(r"charset\s*=\s*[\"']?([^;\s\"']+)", re.I)
ALLOWED_CHARSETS = frozenset(
    {"ascii", "iso-8859-1", "latin-1", "utf-8", "utf8", "windows-1252"}
)


class SourceFetchError(RuntimeError):
    """Base error for a rejected or failed source fetch."""


class ManifestError(SourceFetchError):
    """Raised when the source manifest is malformed or unsafe."""


class UnsafeTargetError(SourceFetchError):
    """Raised when a URL or resolved address violates the fetch policy."""


class ResponseLimitError(SourceFetchError):
    """Raised when a response violates its size or media-type limits."""


@dataclass(frozen=True, slots=True)
class Source:
    id: str
    url: str
    title: str
    domains: tuple[str, ...]
    product: str
    version: str
    license: str
    trust_level: str


@dataclass(frozen=True, slots=True)
class FetchPolicy:
    allowed_hosts: frozenset[str]
    max_response_bytes: int = 8 * 1024 * 1024
    timeout_seconds: float = 30.0
    max_redirects: int = 5


@dataclass(frozen=True, slots=True)
class SourceManifest:
    policy: FetchPolicy
    sources: tuple[Source, ...]

    @property
    def by_id(self) -> dict[str, Source]:
        return {source.id: source for source in self.sources}


@dataclass(frozen=True, slots=True)
class FetchResult:
    source_id: str
    output_path: str
    url: str
    resolved_url: str
    content_sha256: str
    bytes_written: int
    changed: bool


ResolveHost = Callable[[str, int], Awaitable[Sequence[str]]]


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{context} must be a TOML table")
    return value


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field} must be a non-empty string")
    return value.strip()


def _require_positive_int(value: Any, field: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{field} must be an integer")
    if not 0 < value <= maximum:
        raise ManifestError(f"{field} must be between 1 and {maximum}")
    return value


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> SourceManifest:
    manifest_path = Path(path)
    try:
        with manifest_path.open("rb") as source_file:
            raw = tomllib.load(source_file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ManifestError(
            f"cannot read source manifest {manifest_path}: {exc}"
        ) from exc

    if raw.get("manifest_version") != 1:
        raise ManifestError("manifest_version must be 1")

    policy_data = _require_mapping(raw.get("policy"), "policy")
    raw_hosts = policy_data.get("allowed_hosts")
    if not isinstance(raw_hosts, list) or not raw_hosts:
        raise ManifestError("policy.allowed_hosts must be a non-empty list")
    hosts: set[str] = set()
    for index, raw_host in enumerate(raw_hosts):
        host = _require_nonempty_string(
            raw_host, f"policy.allowed_hosts[{index}]"
        ).lower()
        if host != raw_host or host.endswith(".") or "*" in host:
            raise ManifestError(f"invalid canonical host in allowlist: {raw_host!r}")
        if host not in BUILTIN_OFFICIAL_HOSTS:
            raise ManifestError(f"host is not an approved official domain: {host}")
        hosts.add(host)

    timeout = policy_data.get("timeout_seconds", 30)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ManifestError("policy.timeout_seconds must be a number")
    if not 0 < float(timeout) <= 120:
        raise ManifestError("policy.timeout_seconds must be between 0 and 120")
    policy = FetchPolicy(
        allowed_hosts=frozenset(hosts),
        max_response_bytes=_require_positive_int(
            policy_data.get("max_response_bytes", 8 * 1024 * 1024),
            "policy.max_response_bytes",
            32 * 1024 * 1024,
        ),
        timeout_seconds=float(timeout),
        max_redirects=_require_positive_int(
            policy_data.get("max_redirects", 5), "policy.max_redirects", 10
        ),
    )

    raw_sources = raw.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ManifestError("sources must be a non-empty array of tables")
    sources: list[Source] = []
    seen_ids: set[str] = set()
    used_hosts: set[str] = set()
    required_fields = {
        "id",
        "url",
        "title",
        "domains",
        "product",
        "version",
        "license",
        "trust_level",
    }
    for index, raw_source in enumerate(raw_sources):
        entry = _require_mapping(raw_source, f"sources[{index}]")
        missing = sorted(required_fields - entry.keys())
        if missing:
            raise ManifestError(
                f"sources[{index}] is missing required fields: {', '.join(missing)}"
            )
        source_id = _require_nonempty_string(entry["id"], f"sources[{index}].id")
        if not SOURCE_ID_PATTERN.fullmatch(source_id):
            raise ManifestError(f"unsafe source id: {source_id!r}")
        if source_id in seen_ids:
            raise ManifestError(f"duplicate source id: {source_id}")
        seen_ids.add(source_id)

        url = _require_nonempty_string(entry["url"], f"sources[{index}].url")
        host = validate_target_url(url, policy.allowed_hosts)
        used_hosts.add(host)

        raw_domains = entry["domains"]
        if not isinstance(raw_domains, list) or not raw_domains:
            raise ManifestError(f"sources[{index}].domains must be a non-empty list")
        domains: list[str] = []
        for raw_domain in raw_domains:
            domain = _require_nonempty_string(raw_domain, f"sources[{index}].domains")
            if domain not in ALLOWED_DOMAINS:
                raise ManifestError(
                    f"sources[{index}] uses unsupported domain: {domain}"
                )
            if domain not in domains:
                domains.append(domain)

        trust_level = _require_nonempty_string(
            entry["trust_level"], f"sources[{index}].trust_level"
        )
        if trust_level != "official":
            raise ManifestError(
                f"sources[{index}].trust_level must be exactly 'official'"
            )
        sources.append(
            Source(
                id=source_id,
                url=url,
                title=_require_nonempty_string(
                    entry["title"], f"sources[{index}].title"
                ),
                domains=tuple(domains),
                product=_require_nonempty_string(
                    entry["product"], f"sources[{index}].product"
                ),
                version=_require_nonempty_string(
                    entry["version"], f"sources[{index}].version"
                ),
                license=_require_nonempty_string(
                    entry["license"], f"sources[{index}].license"
                ),
                trust_level=trust_level,
            )
        )

    unused_hosts = policy.allowed_hosts - used_hosts
    if unused_hosts:
        raise ManifestError(
            "policy allowlist contains hosts unused by a source: "
            + ", ".join(sorted(unused_hosts))
        )
    return SourceManifest(policy=policy, sources=tuple(sources))


def validate_target_url(url: str, allowed_hosts: Iterable[str]) -> str:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise UnsafeTargetError(f"invalid URL: {url!r}") from exc
    if parsed.scheme != "https":
        raise UnsafeTargetError("only HTTPS source URLs are allowed")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeTargetError("credentials are forbidden in source URLs")
    host = (parsed.hostname or "").lower()
    if not host or host.endswith(".") or host not in frozenset(allowed_hosts):
        raise UnsafeTargetError(
            f"source host is not allowlisted: {host or '<missing>'}"
        )
    if port not in (None, 443):
        raise UnsafeTargetError("only the default HTTPS port is allowed")
    if not parsed.path.startswith("/"):
        raise UnsafeTargetError("source URL must contain an absolute path")
    return host


def ensure_public_addresses(addresses: Iterable[str], host: str) -> tuple[str, ...]:
    checked: list[str] = []
    for raw_address in addresses:
        try:
            address = ipaddress.ip_address(raw_address)
        except ValueError as exc:
            raise UnsafeTargetError(
                f"DNS returned an invalid address for {host}: {raw_address!r}"
            ) from exc
        if not address.is_global:
            raise UnsafeTargetError(
                f"DNS target for {host} is not a public address: {address}"
            )
        checked.append(str(address))
    if not checked:
        raise UnsafeTargetError(f"DNS returned no addresses for {host}")
    return tuple(dict.fromkeys(checked))


async def resolve_public_host(host: str, port: int = 443) -> tuple[str, ...]:
    try:
        records = await asyncio.to_thread(
            socket.getaddrinfo,
            host,
            port,
            socket.AF_UNSPEC,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
        )
    except OSError as exc:
        raise UnsafeTargetError(f"DNS resolution failed for {host}: {exc}") from exc
    addresses = [record[4][0] for record in records]
    return ensure_public_addresses(addresses, host)


class PublicOnlyResolver(AbstractResolver):
    """Validate the addresses actually returned to aiohttp's connector."""

    def __init__(self) -> None:
        self._delegate = aiohttp.DefaultResolver()

    async def resolve(
        self, host: str, port: int = 0, family: int = socket.AF_INET
    ) -> list[dict[str, Any]]:
        records = await self._delegate.resolve(host, port, family)
        ensure_public_addresses((str(record["host"]) for record in records), host)
        return records

    async def close(self) -> None:
        await self._delegate.close()


def _get_header(headers: Mapping[str, str], name: str) -> str | None:
    direct = headers.get(name)
    if direct is not None:
        return str(direct)
    lowered = name.lower()
    for key, value in headers.items():
        if str(key).lower() == lowered:
            return str(value)
    return None


async def _read_limited_response(response: Any, max_bytes: int) -> str:
    content_type_header = _get_header(response.headers, "Content-Type") or ""
    content_type = content_type_header.split(";", maxsplit=1)[0].strip().lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ResponseLimitError(
            f"unsupported response content type: {content_type or '<missing>'}"
        )
    content_length = _get_header(response.headers, "Content-Length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError as exc:
            raise ResponseLimitError("invalid Content-Length header") from exc
        if declared_size < 0 or declared_size > max_bytes:
            raise ResponseLimitError(f"response exceeds {max_bytes} byte limit")

    body = bytearray()
    async for chunk in response.content.iter_chunked(64 * 1024):
        body.extend(chunk)
        if len(body) > max_bytes:
            raise ResponseLimitError(f"response exceeds {max_bytes} byte limit")
    if not body:
        raise ResponseLimitError("source returned an empty document")

    charset_match = CHARSET_PATTERN.search(content_type_header)
    charset = charset_match.group(1).lower() if charset_match else "utf-8"
    if charset not in ALLOWED_CHARSETS:
        raise ResponseLimitError(f"unsupported response charset: {charset}")
    try:
        return bytes(body).decode(charset)
    except UnicodeDecodeError as exc:
        raise ResponseLimitError(f"source body is not valid {charset} text") from exc


def _meta_refresh_target(html: str, base_url: str) -> str | None:
    """Recognize static documentation redirects without executing page scripts."""

    soup = BeautifulSoup(html, "html.parser")
    for meta in soup.find_all("meta"):
        http_equiv = str(meta.get("http-equiv") or "").strip().lower()
        if http_equiv != "refresh":
            continue
        content = str(meta.get("content") or "")
        match = re.fullmatch(
            r"\s*([0-9]+(?:\.[0-9]+)?)\s*;\s*url\s*=\s*['\"]?([^'\"]+)['\"]?\s*",
            content,
            flags=re.IGNORECASE,
        )
        if match is None or float(match.group(1)) > 1.0:
            continue
        return urljoin(base_url, match.group(2).strip())
    return None


async def fetch_html(
    source: Source,
    policy: FetchPolicy,
    session: Any,
    *,
    resolve_host: ResolveHost = resolve_public_host,
) -> tuple[str, str]:
    current_url = source.url
    headers = {
        "Accept": "text/html,application/xhtml+xml;q=0.9",
        "User-Agent": (
            "AL1S-RAG-OfficialDocs/1.0 " "(+https://github.com/SCU-Maker-Org/AL1S)"
        ),
    }
    for redirect_count in range(policy.max_redirects + 1):
        host = validate_target_url(current_url, policy.allowed_hosts)
        await resolve_host(host, 443)
        async with session.get(
            current_url,
            allow_redirects=False,
            headers=headers,
        ) as response:
            if response.status in REDIRECT_STATUSES:
                location = _get_header(response.headers, "Location")
                if not location:
                    raise SourceFetchError("redirect response has no Location header")
                if redirect_count >= policy.max_redirects:
                    raise SourceFetchError("source exceeded the redirect limit")
                current_url = urljoin(current_url, location)
                validate_target_url(current_url, policy.allowed_hosts)
                continue
            if response.status != 200:
                raise SourceFetchError(f"source returned HTTP {response.status}")
            html = await _read_limited_response(
                response, max_bytes=policy.max_response_bytes
            )
            meta_target = _meta_refresh_target(html, current_url)
            if meta_target is not None:
                if redirect_count >= policy.max_redirects:
                    raise SourceFetchError("source exceeded the redirect limit")
                validate_target_url(meta_target, policy.allowed_hosts)
                current_url = meta_target
                continue
            return html, current_url
    raise SourceFetchError("source exceeded the redirect limit")


def _normalize_inline(text: str) -> str:
    lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _inline_markdown(node: Any, base_url: str) -> str:
    if isinstance(node, NavigableString):
        return str(node)
    if not isinstance(node, Tag):
        return ""
    name = node.name.lower()
    if name == "br":
        return "\n"
    if name == "code":
        content = _normalize_inline(node.get_text(" ", strip=True))
        if not content:
            return ""
        fence = "``" if "`" in content else "`"
        return f"{fence}{content}{fence}"
    rendered = _normalize_inline(
        "".join(_inline_markdown(child, base_url) for child in node.children)
    )
    if not rendered:
        return ""
    if name in {"strong", "b"}:
        return f"**{rendered}**"
    if name in {"em", "i"}:
        return f"*{rendered}*"
    if name == "a":
        href = str(node.get("href") or "").strip()
        if not href or href.startswith(("#", "mailto:", "javascript:", "data:")):
            return rendered
        target = urljoin(base_url, href)
        parsed = urlsplit(target)
        if parsed.scheme not in {"http", "https"} or parsed.username is not None:
            return rendered
        target = target.replace(" ", "%20").replace(")", "%29")
        return f"[{rendered}]({target})"
    if name == "img":
        alt = _normalize_inline(str(node.get("alt") or ""))
        return alt
    return rendered


def _render_list(node: Tag, base_url: str, depth: int = 0) -> str:
    ordered = node.name.lower() == "ol"
    lines: list[str] = []
    items = node.find_all("li", recursive=False)
    for index, item in enumerate(items, start=1):
        inline_parts = [
            _inline_markdown(child, base_url)
            for child in item.children
            if not (isinstance(child, Tag) and child.name.lower() in {"ul", "ol"})
        ]
        value = _normalize_inline("".join(inline_parts))
        marker = f"{index}." if ordered else "-"
        indent = "  " * depth
        if value:
            lines.append(f"{indent}{marker} {value}")
        for nested in item.find_all(["ul", "ol"], recursive=False):
            nested_text = _render_list(nested, base_url, depth + 1)
            if nested_text:
                lines.append(nested_text)
    return "\n".join(lines)


def _render_table(node: Tag, base_url: str) -> str:
    rows: list[list[str]] = []
    has_header = False
    for row in node.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        has_header = has_header or any(cell.name.lower() == "th" for cell in cells)
        rows.append(
            [
                _normalize_inline(_inline_markdown(cell, base_url)).replace("|", "\\|")
                for cell in cells
            ]
        )
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    if not has_header:
        rows.insert(0, [f"Column {index}" for index in range(1, width + 1)])
    output = ["| " + " | ".join(rows[0]) + " |"]
    output.append("| " + " | ".join("---" for _ in range(width)) + " |")
    output.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(output)


def _render_blocks(node: Tag, base_url: str) -> list[str]:
    blocks: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            text = _normalize_inline(str(child))
            if text:
                blocks.append(text)
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name.lower()
        if re.fullmatch(r"h[1-6]", name):
            title = _normalize_inline(_inline_markdown(child, base_url))
            if title:
                blocks.append(f"{'#' * int(name[1])} {title}")
        elif name == "p":
            paragraph = _normalize_inline(_inline_markdown(child, base_url))
            if paragraph:
                blocks.append(paragraph)
        elif name == "pre":
            code = child.get_text("", strip=False).strip("\n")
            if code:
                code_node = child.find("code")
                language = ""
                if code_node is not None:
                    classes = code_node.get("class") or []
                    language_class = next(
                        (item for item in classes if str(item).startswith("language-")),
                        "",
                    )
                    language = str(language_class).removeprefix("language-")
                fence = "```" if "```" not in code else "````"
                blocks.append(f"{fence}{language}\n{code}\n{fence}")
        elif name in {"ul", "ol"}:
            rendered = _render_list(child, base_url)
            if rendered:
                blocks.append(rendered)
        elif name == "table":
            rendered = _render_table(child, base_url)
            if rendered:
                blocks.append(rendered)
        elif name == "blockquote":
            rendered = "\n\n".join(_render_blocks(child, base_url))
            if rendered:
                blocks.append("\n".join(f"> {line}" for line in rendered.splitlines()))
        elif name == "dl":
            definitions: list[str] = []
            for entry in child.find_all(["dt", "dd"], recursive=False):
                value = _normalize_inline(_inline_markdown(entry, base_url))
                if not value:
                    continue
                definitions.append(
                    f"**{value}**" if entry.name.lower() == "dt" else value
                )
            if definitions:
                blocks.append("\n\n".join(definitions))
        elif name == "hr":
            blocks.append("---")
        else:
            nested_blocks = _render_blocks(child, base_url)
            if nested_blocks:
                blocks.extend(nested_blocks)
            else:
                inline = _normalize_inline(_inline_markdown(child, base_url))
                if inline:
                    blocks.append(inline)
    return blocks


def html_to_markdown(html: str, base_url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for unwanted in soup.find_all(
        [
            "script",
            "style",
            "noscript",
            "template",
            "nav",
            "footer",
            "aside",
            "form",
            "dialog",
        ]
    ):
        unwanted.decompose()
    main = (
        soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find("article")
        or soup.find(id="content")
        or soup.find(class_="document")
        or soup.find(class_="content")
        or soup.body
    )
    if main is None:
        raise SourceFetchError("HTML document has no readable body")
    markdown = "\n\n".join(_render_blocks(main, base_url))
    markdown = re.sub(r"\n[ \t]+\n", "\n\n", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    if not markdown:
        raise SourceFetchError("HTML document has no readable main content")
    return markdown + "\n"


def build_markdown_document(
    source: Source,
    markdown_body: str,
    *,
    resolved_url: str | None = None,
    fetched_at: datetime | None = None,
) -> tuple[str, str]:
    body = markdown_body.strip() + "\n"
    content_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()
    timestamp = fetched_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    metadata = {
        "source_id": source.id,
        "source_uri": source.url,
        "url": source.url,
        "resolved_url": resolved_url or source.url,
        "title": source.title,
        "domains": list(source.domains),
        "product": source.product,
        "version": source.version,
        "license": source.license,
        "trust_level": source.trust_level,
        "fetched_at": timestamp.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "content_sha256": content_sha256,
    }
    frontmatter = yaml.safe_dump(
        metadata,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    ).strip()
    return f"---\n{frontmatter}\n---\n\n{body}", content_sha256


def existing_document_matches(
    path: str | Path,
    source: Source,
    content_sha256: str,
    *,
    max_bytes: int,
) -> bool:
    """Verify both metadata and body before treating a previous fetch as unchanged."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        return False
    try:
        with candidate.open("rb") as existing_file:
            raw = existing_file.read(max_bytes + 1)
    except OSError:
        return False
    if len(raw) > max_bytes or b"\x00" in raw:
        return False
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return False
    match = re.match(r"\A---\r?\n(.*?)\r?\n---\r?\n", text, flags=re.DOTALL)
    if match is None:
        return False
    try:
        metadata = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return False
    if not isinstance(metadata, Mapping):
        return False
    body = text[match.end() :]
    if body.startswith(("\r\n", "\n")):
        body = body.removeprefix("\r\n").removeprefix("\n")
    actual_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return (
        metadata.get("source_id") == source.id
        and metadata.get("source_uri") == source.url
        and metadata.get("url") == source.url
        and metadata.get("title") == source.title
        and metadata.get("domains") == list(source.domains)
        and metadata.get("product") == source.product
        and metadata.get("version") == source.version
        and metadata.get("license") == source.license
        and metadata.get("trust_level") == source.trust_level
        and metadata.get("content_sha256") == content_sha256
        and actual_hash == content_sha256
    )


def atomic_write(path: str | Path, content: str) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = content.encode("utf-8")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
        try:
            directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    except Exception:
        try:
            os.close(file_descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return len(payload)


async def fetch_source(
    source: Source,
    policy: FetchPolicy,
    session: Any,
    output_directory: str | Path,
    *,
    resolve_host: ResolveHost = resolve_public_host,
    fetched_at: datetime | None = None,
) -> FetchResult:
    html, resolved_url = await fetch_html(
        source, policy, session, resolve_host=resolve_host
    )
    markdown_body = html_to_markdown(html, resolved_url)
    document, content_sha256 = build_markdown_document(
        source,
        markdown_body,
        resolved_url=resolved_url,
        fetched_at=fetched_at,
    )
    output_path = Path(output_directory) / f"{source.id}.md"
    max_existing_bytes = policy.max_response_bytes + 128 * 1024
    unchanged = await asyncio.to_thread(
        existing_document_matches,
        output_path,
        source,
        content_sha256,
        max_bytes=max_existing_bytes,
    )
    bytes_written = (
        0 if unchanged else await asyncio.to_thread(atomic_write, output_path, document)
    )
    return FetchResult(
        source_id=source.id,
        output_path=str(output_path),
        url=source.url,
        resolved_url=resolved_url,
        content_sha256=content_sha256,
        bytes_written=bytes_written,
        changed=not unchanged,
    )


async def fetch_selected_sources(
    manifest: SourceManifest,
    selected: Sequence[Source],
    output_directory: str | Path,
    *,
    session: Any | None = None,
    resolve_host: ResolveHost = resolve_public_host,
) -> tuple[list[FetchResult], list[dict[str, str]]]:
    owned_session = session is None
    resolver: PublicOnlyResolver | None = None
    if session is None:
        resolver = PublicOnlyResolver()
        connector = aiohttp.TCPConnector(
            resolver=resolver,
            use_dns_cache=False,
            ssl=True,
        )
        timeout = aiohttp.ClientTimeout(
            total=manifest.policy.timeout_seconds,
            connect=min(10.0, manifest.policy.timeout_seconds),
        )
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            trust_env=False,
            cookie_jar=aiohttp.DummyCookieJar(),
        )

    semaphore = asyncio.Semaphore(4)

    async def run_one(source: Source) -> FetchResult:
        async with semaphore:
            return await fetch_source(
                source,
                manifest.policy,
                session,
                output_directory,
                resolve_host=resolve_host,
            )

    try:
        settled = await asyncio.gather(
            *(run_one(source) for source in selected), return_exceptions=True
        )
    finally:
        if owned_session:
            await session.close()
            if resolver is not None:
                await resolver.close()

    results: list[FetchResult] = []
    errors: list[dict[str, str]] = []
    for source, value in zip(selected, settled, strict=True):
        if isinstance(value, BaseException):
            errors.append(
                {
                    "source_id": source.id,
                    "error": f"{type(value).__name__}: {value}",
                }
            )
        else:
            results.append(value)
    return results, errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch allowlisted official documentation into the AL1S RAG corpus."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--source",
        action="append",
        metavar="ID",
        help="fetch one manifest source ID; repeat to select multiple sources",
    )
    selection.add_argument("--all", action="store_true", help="fetch every source")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    return parser


async def _async_main(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    by_id = manifest.by_id
    if args.all:
        selected = list(manifest.sources)
    else:
        requested = list(dict.fromkeys(args.source or []))
        unknown = sorted(set(requested) - by_id.keys())
        if unknown:
            raise ManifestError(f"unknown source id(s): {', '.join(unknown)}")
        selected = [by_id[source_id] for source_id in requested]

    results, errors = await fetch_selected_sources(
        manifest, selected, args.output.resolve()
    )
    report = {
        "ok": len(results),
        "failed": len(errors),
        "output": str(args.output.resolve()),
        "results": [asdict(result) for result in results],
        "errors": errors,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        for result in results:
            if result.changed:
                print(
                    f"fetched {result.source_id}: {result.output_path} "
                    f"({result.bytes_written} bytes)"
                )
            else:
                print(f"unchanged {result.source_id}: {result.output_path}")
        for error in errors:
            print(
                f"failed {error['source_id']}: {error['error']}",
                file=sys.stderr,
            )
        print(f"completed: {len(results)} fetched, {len(errors)} failed")
    return 1 if errors else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_async_main(args))
    except (ManifestError, UnsafeTargetError) as exc:
        if args.json:
            print(
                json.dumps(
                    {"ok": 0, "failed": 1, "error": str(exc)},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
