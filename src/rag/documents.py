from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import yaml
from bs4 import BeautifulSoup

Domain = Literal[
    "sys",
    "hpc",
    "compile",
    "distributed",
    "db",
    "storage",
    "ai_workload",
    "cloud",
    "security",
]

DOMAIN_ORDER: tuple[Domain, ...] = (
    "sys",
    "hpc",
    "compile",
    "distributed",
    "db",
    "storage",
    "ai_workload",
    "cloud",
    "security",
)
ALLOWED_DOMAINS = frozenset(DOMAIN_ORDER)

SUPPORTED_EXTENSIONS = frozenset(
    {
        ".asm",
        ".bash",
        ".c",
        ".cc",
        ".cfg",
        ".cmake",
        ".conf",
        ".cpp",
        ".cs",
        ".css",
        ".cu",
        ".cuh",
        ".cxx",
        ".dart",
        ".erl",
        ".ex",
        ".exs",
        ".fish",
        ".fs",
        ".fsx",
        ".go",
        ".gradle",
        ".groovy",
        ".h",
        ".hh",
        ".hpp",
        ".hrl",
        ".htm",
        ".html",
        ".hip",
        ".ini",
        ".ipynb",
        ".java",
        ".jl",
        ".js",
        ".json",
        ".jsx",
        ".kt",
        ".kts",
        ".less",
        ".ll",
        ".lhs",
        ".lua",
        ".md",
        ".markdown",
        ".mdown",
        ".php",
        ".proto",
        ".ps1",
        ".py",
        ".pyi",
        ".r",
        ".rb",
        ".rs",
        ".rst",
        ".s",
        ".scala",
        ".scss",
        ".sh",
        ".sql",
        ".svelte",
        ".text",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".zig",
        ".vue",
        ".xml",
        ".yaml",
        ".yml",
        ".zsh",
    }
)
SUPPORTED_FILENAMES = frozenset(
    {
        "build",
        "cmakelists.txt",
        "containerfile",
        "dockerfile",
        "gemfile",
        "justfile",
        "license",
        "makefile",
        "meson.build",
        "procfile",
        "rakefile",
        "requirements.txt",
        "readme",
        "vagrantfile",
        "workspace",
    }
)

_MARKDOWN_EXTENSIONS = frozenset({".md", ".markdown", ".mdown"})
_HTML_EXTENSIONS = frozenset({".html", ".htm"})
_STRUCTURED_EXTENSIONS = frozenset({".json", ".toml", ".yaml", ".yml"})
_FRONTMATTER_BOUNDARY = re.compile(r"^---[ \t]*$")
_ATX_TITLE = re.compile(r"(?m)^#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$")


class DocumentError(ValueError):
    """Base exception for rejected RAG source documents."""


class UnsupportedDocumentError(DocumentError):
    """Raised when a path is not a supported text document."""


class InvalidDomainError(DocumentError):
    """Raised when metadata contains a domain outside the controlled vocabulary."""


class UnsafeDocumentPathError(DocumentError):
    """Raised when a document resolves outside its allowed root."""


class DocumentTooLargeError(DocumentError):
    """Raised when a document exceeds the configured byte limit."""


class DocumentDecodeError(DocumentError):
    """Raised when a source is binary or is not valid UTF-8 text."""


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    source_path: str
    namespace: str
    content: str
    content_sha256: str
    source_sha256: str
    revision_sha256: str
    byte_size: int
    format: str
    domains: tuple[Domain, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    source_path: str
    ordinal: int
    content: str
    content_sha256: str
    start_char: int
    end_char: int
    heading_path: tuple[str, ...] = ()
    domains: tuple[Domain, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


def stable_sha256(*parts: str | bytes) -> str:
    """Hash length-delimited values so IDs remain deterministic and unambiguous."""

    digest = hashlib.sha256()
    for part in parts:
        value = part if isinstance(part, bytes) else part.encode("utf-8")
        digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
        digest.update(value)
    return digest.hexdigest()


def normalize_domains(values: str | Iterable[str] | None) -> tuple[Domain, ...]:
    if values is None:
        return ()
    candidates = values.split(",") if isinstance(values, str) else list(values)
    normalized = {
        str(value).strip().lower() for value in candidates if str(value).strip()
    }
    invalid = sorted(normalized - ALLOWED_DOMAINS)
    if invalid:
        allowed = ", ".join(DOMAIN_ORDER)
        raise InvalidDomainError(
            f"Unsupported domain label(s): {', '.join(invalid)}. Allowed: {allowed}"
        )
    return tuple(domain for domain in DOMAIN_ORDER if domain in normalized)


def is_supported_document(path: str | Path) -> bool:
    candidate = Path(path)
    filename = candidate.name.lower()
    return (
        candidate.suffix.lower() in SUPPORTED_EXTENSIONS
        or filename in SUPPORTED_FILENAMES
        or filename.startswith(("dockerfile.", "containerfile."))
    )


def document_format(path: str | Path) -> str:
    candidate = Path(path)
    suffix = candidate.suffix.lower()
    if suffix in _MARKDOWN_EXTENSIONS:
        return "markdown"
    if suffix in _HTML_EXTENSIONS:
        return "html"
    if suffix == ".rst":
        return "rst"
    if suffix in {".txt", ".text"}:
        return "text"
    if suffix in _STRUCTURED_EXTENSIONS:
        return suffix.removeprefix(".")
    if candidate.name.lower() in {"dockerfile", "containerfile"}:
        return "dockerfile"
    if candidate.name.lower() in {"makefile", "rakefile"}:
        return "makefile"
    return suffix.removeprefix(".") or candidate.name.lower()


def load_document(
    path: str | Path,
    *,
    root: str | Path | None = None,
    namespace: str = "default",
    default_domains: str | Iterable[str] | None = None,
    base_metadata: Mapping[str, Any] | None = None,
    max_file_bytes: int = 8 * 1024 * 1024,
) -> Document:
    if max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be greater than zero")
    if not namespace.strip():
        raise ValueError("namespace must not be empty")

    candidate = Path(path)
    if not is_supported_document(candidate):
        raise UnsupportedDocumentError(f"Unsupported document type: {candidate.name}")

    resolved = candidate.resolve(strict=True)
    allowed_root = (
        Path(root).resolve(strict=True) if root is not None else resolved.parent
    )
    if allowed_root.is_file():
        allowed_root = allowed_root.parent
    if not resolved.is_relative_to(allowed_root):
        raise UnsafeDocumentPathError(
            f"Document resolves outside the allowed root: {candidate}"
        )
    if not resolved.is_file():
        raise UnsupportedDocumentError(f"Document is not a regular file: {candidate}")

    stat_size = resolved.stat().st_size
    if stat_size > max_file_bytes:
        raise DocumentTooLargeError(
            f"Document exceeds {max_file_bytes} bytes: {candidate}"
        )
    with resolved.open("rb") as source:
        raw = source.read(max_file_bytes + 1)
    if len(raw) > max_file_bytes:
        raise DocumentTooLargeError(
            f"Document exceeds {max_file_bytes} bytes: {candidate}"
        )
    if b"\x00" in raw:
        raise DocumentDecodeError(f"Binary document rejected: {candidate}")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentDecodeError(f"Document is not valid UTF-8: {candidate}") from exc

    source_path = _relative_source_path(candidate, allowed_root)
    fmt = document_format(candidate)
    metadata = _normalize_metadata(base_metadata or {})
    frontmatter: dict[str, Any] = {}
    if candidate.suffix.lower() not in _STRUCTURED_EXTENSIONS | _HTML_EXTENSIONS:
        frontmatter, text = _extract_frontmatter(text, candidate)
        metadata.update(frontmatter)

    if fmt == "html":
        text, html_metadata = _html_to_text(text)
        for key, value in html_metadata.items():
            metadata.setdefault(key, value)

    metadata.setdefault("title", _infer_title(text, candidate, fmt))
    frontmatter_domains = metadata.pop("domain", None)
    plural_domains = metadata.get("domains")
    declared_domains = [
        *_coerce_domain_values(frontmatter_domains),
        *_coerce_domain_values(plural_domains),
    ]
    domains = normalize_domains(
        declared_domains if declared_domains else default_domains
    )
    metadata["domains"] = list(domains)
    metadata = _normalize_metadata(metadata)

    content = text.strip()
    content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    source_sha256 = hashlib.sha256(raw).hexdigest()
    metadata_json = json.dumps(
        metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    revision_sha256 = stable_sha256(
        "revision",
        source_sha256,
        fmt,
        metadata_json,
        ",".join(domains),
    )
    return Document(
        document_id=stable_sha256("document", namespace.strip(), source_path),
        source_path=source_path,
        namespace=namespace.strip(),
        content=content,
        content_sha256=content_sha256,
        source_sha256=source_sha256,
        revision_sha256=revision_sha256,
        byte_size=len(raw),
        format=fmt,
        domains=domains,
        metadata=metadata,
    )


def _relative_source_path(candidate: Path, allowed_root: Path) -> str:
    logical = candidate.absolute()
    try:
        relative = logical.relative_to(allowed_root.absolute())
    except ValueError:
        relative = candidate.resolve(strict=True).relative_to(allowed_root)
    source = relative.as_posix()
    if not source or source == "." or source.startswith("../"):
        raise UnsafeDocumentPathError(f"Invalid relative document path: {candidate}")
    return source


def _extract_frontmatter(text: str, path: Path) -> tuple[dict[str, Any], str]:
    lines = text.splitlines(keepends=True)
    if not lines or not _FRONTMATTER_BOUNDARY.fullmatch(lines[0].rstrip("\r\n")):
        return {}, text
    closing_index = next(
        (
            index
            for index, line in enumerate(lines[1:], start=1)
            if _FRONTMATTER_BOUNDARY.fullmatch(line.rstrip("\r\n"))
        ),
        None,
    )
    if closing_index is None:
        return {}, text
    header = "".join(lines[1:closing_index])
    try:
        parsed = yaml.safe_load(header) or {}
    except yaml.YAMLError as exc:
        raise DocumentError(f"Invalid YAML frontmatter in {path}") from exc
    if not isinstance(parsed, dict):
        raise DocumentError(f"YAML frontmatter must be a mapping in {path}")
    return _normalize_metadata(parsed), "".join(lines[closing_index + 1 :]).lstrip(
        "\r\n"
    )


def _html_to_text(text: str) -> tuple[str, dict[str, Any]]:
    soup = BeautifulSoup(text, "html.parser")
    metadata: dict[str, Any] = {}
    if soup.title and soup.title.get_text(strip=True):
        metadata["title"] = soup.title.get_text(" ", strip=True)
    if soup.head:
        soup.head.decompose()
    for element in soup(["script", "style", "noscript", "template"]):
        element.decompose()
    for level in range(1, 7):
        for heading in soup.find_all(f"h{level}"):
            heading.replace_with(
                f"\n\n{'#' * level} {heading.get_text(' ', strip=True)}\n\n"
            )
    for line_break in soup.find_all("br"):
        line_break.replace_with("\n")
    fragments = [fragment.strip() for fragment in soup.get_text("\n").splitlines()]
    normalized = "\n".join(fragment for fragment in fragments if fragment)
    return normalized, metadata


def _infer_title(text: str, path: Path, fmt: str) -> str:
    if fmt in {"markdown", "html"}:
        match = _ATX_TITLE.search(text)
        if match:
            return match.group(1).strip()
    if fmt == "rst":
        lines = text.splitlines()
        for index in range(len(lines) - 1):
            title = lines[index].strip()
            underline = lines[index + 1].strip()
            if title and len(underline) >= 3 and len(set(underline)) == 1:
                if underline[0] in "=-~^\"'+*#`:_<>":
                    return title
    return path.stem or path.name


def _coerce_domain_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, Mapping)):
        return [str(part).strip() for part in value if str(part).strip()]
    raise InvalidDomainError("domain/domains metadata must be a string or list")


def _normalize_metadata(value: Mapping[Any, Any]) -> dict[str, Any]:
    return {str(key): _normalize_metadata_value(item) for key, item in value.items()}


def _normalize_metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return _normalize_metadata(value)
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_normalize_metadata_value(item) for item in value),
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
        )
    if isinstance(value, (list, tuple)):
        return [_normalize_metadata_value(item) for item in value]
    return str(value)
