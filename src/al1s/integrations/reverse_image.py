"""Direct trace.moe and SauceNAO reverse image APIs with bounded downloads."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import aiohttp
import puremagic
from aiohttp.abc import AbstractResolver

from al1s.config import AppSettings, SecretStore
from al1s.ports.search import ImageSearchHit

_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_REDIRECTS = {301, 302, 303, 307, 308}


class ReverseImageError(RuntimeError):
    pass


def _ensure_public(addresses: list[str]) -> None:
    if not addresses:
        raise ReverseImageError("DNS returned no addresses")
    for raw in addresses:
        try:
            address = ipaddress.ip_address(raw)
        except ValueError as exc:
            raise ReverseImageError("DNS returned an invalid address") from exc
        if not address.is_global:
            raise ReverseImageError("URL resolved to a non-public address")


class _PublicResolver(AbstractResolver):
    def __init__(self) -> None:
        self._delegate = aiohttp.DefaultResolver()

    async def resolve(
        self, host: str, port: int = 0, family: int = socket.AF_INET
    ) -> list[dict[str, Any]]:
        records = await self._delegate.resolve(host, port, family)
        _ensure_public([str(record["host"]) for record in records])
        return records

    async def close(self) -> None:
        await self._delegate.close()


def _safe_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ReverseImageError("invalid image URL") from exc
    if parsed.scheme != "https" or port not in {None, 443}:
        raise ReverseImageError("image URL must use HTTPS")
    if parsed.username or parsed.password or not parsed.hostname or parsed.fragment:
        raise ReverseImageError("image URL contains forbidden components")
    try:
        literal = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        literal = None
    if literal is not None and not literal.is_global:
        raise ReverseImageError("image URL targets a non-public address")
    return urlunsplit(parsed)


class ReverseImageService:
    def __init__(self, settings: AppSettings, secrets: SecretStore):
        self.settings = settings.reverse_image
        self.secrets = secrets
        self.saucenao_key = (
            secrets.read(self.settings.saucenao_api_key_secret, required=True)
            if self.settings.saucenao_enabled
            else ""
        )
        connector = aiohttp.TCPConnector(resolver=_PublicResolver(), ttl_dns_cache=0)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=45),
            cookie_jar=aiohttp.DummyCookieJar(),
        )

    async def search_url(self, url: str) -> tuple[ImageSearchHit, ...]:
        data, mime_type = await self._download(url)
        return await self.search_bytes(data, mime_type)

    async def search_bytes(self, data: bytes, mime_type: str) -> tuple[ImageSearchHit, ...]:
        if not data or len(data) > self.settings.max_image_bytes:
            raise ReverseImageError("image is empty or exceeds the configured size")
        try:
            detected = puremagic.from_string(data, mime=True)
        except puremagic.PureError as exc:
            raise ReverseImageError("unsupported image format") from exc
        if detected not in _IMAGE_TYPES or mime_type not in _IMAGE_TYPES:
            raise ReverseImageError("unsupported or mismatched image format")
        calls = []
        if self.settings.trace_moe_enabled:
            calls.append(self._trace_moe(data, detected))
        if self.settings.saucenao_enabled:
            calls.append(self._saucenao(data, detected))
        if not calls:
            return ()
        results = await asyncio.gather(*calls, return_exceptions=True)
        hits: list[ImageSearchHit] = []
        errors: list[str] = []
        succeeded = 0
        for result in results:
            if isinstance(result, tuple):
                succeeded += 1
                hits.extend(result)
            elif isinstance(result, BaseException):
                errors.append(type(result).__name__)
        if calls and succeeded == 0:
            raise ReverseImageError("all reverse image providers failed: " + ", ".join(errors))
        hits.sort(key=lambda hit: hit.similarity or 0.0, reverse=True)
        return tuple(hits[: self.settings.max_results])

    async def _download(self, value: str) -> tuple[bytes, str]:
        current = _safe_url(value)
        for redirect in range(4):
            async with self.session.get(current, allow_redirects=False) as response:
                if response.status in _REDIRECTS:
                    location = response.headers.get("Location")
                    if not location or redirect == 3:
                        raise ReverseImageError("image redirect was rejected")
                    current = _safe_url(urljoin(current, location))
                    continue
                if response.status != 200:
                    raise ReverseImageError(f"image download failed with HTTP {response.status}")
                mime_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
                if mime_type not in _IMAGE_TYPES:
                    raise ReverseImageError("URL did not return a supported image")
                return await self._read_limited(response), mime_type
        raise ReverseImageError("image redirect was rejected")

    async def _read_limited(self, response: aiohttp.ClientResponse) -> bytes:
        limit = self.settings.max_image_bytes
        if response.content_length is not None and response.content_length > limit:
            raise ReverseImageError("image exceeds the configured size")
        body = bytearray()
        async for chunk in response.content.iter_chunked(64 * 1024):
            body.extend(chunk)
            if len(body) > limit:
                raise ReverseImageError("image exceeds the configured size")
        return bytes(body)

    async def _trace_moe(self, data: bytes, mime_type: str) -> tuple[ImageSearchHit, ...]:
        form = aiohttp.FormData()
        form.add_field("image", data, filename="image", content_type=mime_type)
        async with self.session.post(
            "https://api.trace.moe/search", data=form, allow_redirects=False
        ) as response:
            if response.status != 200:
                raise ReverseImageError(f"trace.moe returned HTTP {response.status}")
            payload = await self._read_json(response)
        if not isinstance(payload, dict):
            raise ReverseImageError("trace.moe returned invalid JSON")
        hits = []
        for item in payload.get("result", [])[: self.settings.max_results]:
            anime_id = item.get("anilist")
            title = str(item.get("filename") or f"AniList {anime_id}")
            hits.append(
                ImageSearchHit(
                    engine="trace.moe",
                    title=title,
                    url=f"https://anilist.co/anime/{anime_id}" if anime_id else "https://trace.moe",
                    similarity=float(item.get("similarity", 0.0)),
                    preview_url=item.get("image"),
                    metadata={
                        "episode": item.get("episode"),
                        "from": item.get("from"),
                        "to": item.get("to"),
                    },
                )
            )
        return tuple(hits)

    async def _saucenao(self, data: bytes, mime_type: str) -> tuple[ImageSearchHit, ...]:
        form = aiohttp.FormData()
        form.add_field("api_key", self.saucenao_key)
        form.add_field("output_type", "2")
        form.add_field("db", "999")
        form.add_field("numres", str(self.settings.max_results))
        form.add_field("file", data, filename="image", content_type=mime_type)
        async with self.session.post(
            "https://saucenao.com/search.php", data=form, allow_redirects=False
        ) as response:
            if response.status != 200:
                raise ReverseImageError(f"SauceNAO returned HTTP {response.status}")
            payload = await self._read_json(response)
        if not isinstance(payload, dict):
            raise ReverseImageError("SauceNAO returned invalid JSON")
        hits = []
        for item in payload.get("results", [])[: self.settings.max_results]:
            data_item = item.get("data") or {}
            header = item.get("header") or {}
            urls = data_item.get("ext_urls") or []
            if not urls:
                continue
            title = str(
                data_item.get("title")
                or data_item.get("material")
                or data_item.get("source")
                or "SauceNAO result"
            )
            try:
                similarity = float(header.get("similarity", 0.0)) / 100
            except (TypeError, ValueError):
                similarity = None
            hits.append(
                ImageSearchHit(
                    engine="SauceNAO",
                    title=title,
                    url=str(urls[0]),
                    similarity=similarity,
                    preview_url=header.get("thumbnail"),
                )
            )
        return tuple(hits)

    async def health(self) -> dict[str, str]:
        status: dict[str, str] = {}
        targets = {}
        if self.settings.trace_moe_enabled:
            targets["trace.moe"] = "https://api.trace.moe/me"
        if self.settings.saucenao_enabled:
            targets["SauceNAO"] = "https://saucenao.com/"
        for name, url in targets.items():
            try:
                async with self.session.get(url, allow_redirects=False) as response:
                    status[name] = "ok" if response.status < 500 else f"HTTP {response.status}"
            except (aiohttp.ClientError, TimeoutError):
                status[name] = "unreachable"
        return status

    @staticmethod
    async def _read_json(response: aiohttp.ClientResponse) -> object:
        limit = 2 * 1024 * 1024
        if response.content_length is not None and response.content_length > limit:
            raise ReverseImageError("reverse image provider response is too large")
        body = bytearray()
        async for chunk in response.content.iter_chunked(64 * 1024):
            body.extend(chunk)
            if len(body) > limit:
                raise ReverseImageError("reverse image provider response is too large")
        try:
            return json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReverseImageError("reverse image provider returned invalid JSON") from exc

    async def close(self) -> None:
        await self.session.close()
