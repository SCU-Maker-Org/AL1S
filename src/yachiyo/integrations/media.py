"""Direct bounded DashScope image and speech generation."""

from __future__ import annotations

import ipaddress
import json
import re
from typing import Any
from urllib.parse import urljoin, urlsplit

import aiohttp

from yachiyo.config import AppSettings, SecretStore
from yachiyo.domain import Artifact
from yachiyo.domain.content import ArtifactKind

from .reverse_image import _PublicResolver

_OFFICIAL_HOST = re.compile(r"(?:[a-z0-9-]+\.)*dashscope\.aliyuncs\.com", re.I)
_ARTIFACT_HOST = re.compile(r"(?:[a-z0-9-]+\.)+aliyuncs\.com", re.I)
_REDIRECTS = {301, 302, 303, 307, 308}


class MediaError(RuntimeError):
    pass


class MediaService:
    def __init__(self, settings: AppSettings, secrets: SecretStore):
        self.settings = settings.media
        self.api_key = secrets.read(self.settings.api_key_secret, required=True)
        parsed = urlsplit(str(self.settings.base_url))
        if (
            parsed.scheme != "https"
            or parsed.port not in {None, 443}
            or not parsed.hostname
            or not _OFFICIAL_HOST.fullmatch(parsed.hostname)
            or parsed.username
            or parsed.password
            or parsed.path.rstrip("/") != "/api/v1"
            or parsed.query
            or parsed.fragment
        ):
            raise MediaError("media base URL must be an official DashScope HTTPS endpoint")
        self.base_url = str(self.settings.base_url).rstrip("/")
        connector = aiohttp.TCPConnector(resolver=_PublicResolver(), ttl_dns_cache=0)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=120),
            cookie_jar=aiohttp.DummyCookieJar(),
        )

    async def generate_image(self, prompt: str, *, size: str = "1024x1024") -> Artifact:
        prompt = self._text(prompt, 4000)
        match = re.fullmatch(r"(\d{3,4})x(\d{3,4})", size)
        if match is None:
            raise MediaError("image size must use WIDTHxHEIGHT")
        width, height = map(int, match.groups())
        if not 512 * 512 <= width * height <= 2048 * 2048:
            raise MediaError("image dimensions are outside the allowed range")
        response = await self._post_json(
            "services/aigc/multimodal-generation/generation",
            {
                "model": self.settings.image_model,
                "input": {
                    "messages": [
                        {"role": "user", "content": [{"text": prompt}]},
                    ]
                },
                "parameters": {
                    "size": size.replace("x", "*"),
                    "n": 1,
                    "prompt_extend": True,
                    "watermark": False,
                },
            },
        )
        try:
            content = response["output"]["choices"][0]["message"]["content"]
            url = next(
                item["image"]
                for item in content
                if isinstance(item, dict) and isinstance(item.get("image"), str)
            )
        except (KeyError, IndexError, StopIteration, TypeError) as exc:
            raise MediaError("image provider returned no artifact") from exc
        data = await self._download(url, {"image/png", "application/octet-stream"})
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise MediaError("image provider returned invalid PNG data")
        return Artifact(ArtifactKind.IMAGE, "yachiyo.png", "image/png", data=data)

    async def synthesize_speech(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str = "opus",
        speed: float = 1.0,
    ) -> Artifact:
        text = self._text(text, 10_000)
        if response_format not in {"opus", "mp3"}:
            raise MediaError("speech format must be opus or mp3")
        if not 0.5 <= speed <= 2.0:
            raise MediaError("speech speed must be between 0.5 and 2.0")
        response = await self._post_json(
            "services/audio/tts/SpeechSynthesizer",
            {
                "model": self.settings.speech_model,
                "input": {
                    "text": text,
                    "voice": voice or self.settings.speech_voice,
                    "format": response_format,
                    "sample_rate": 24_000,
                    "rate": speed,
                    "enable_aigc_tag": True,
                },
            },
        )
        try:
            url = response["output"]["audio"]["url"]
        except (KeyError, TypeError) as exc:
            raise MediaError("speech provider returned no artifact") from exc
        content_types = (
            {"audio/ogg", "audio/opus", "application/ogg", "application/octet-stream"}
            if response_format == "opus"
            else {"audio/mpeg", "audio/mp3", "application/octet-stream"}
        )
        data = await self._download(url, content_types)
        valid = (
            data.startswith(b"OggS") and b"OpusHead" in data[:256]
            if response_format == "opus"
            else data.startswith(b"ID3")
            or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0)
        )
        if not valid:
            raise MediaError("speech provider returned invalid audio data")
        mime = "audio/ogg" if response_format == "opus" else "audio/mpeg"
        return Artifact(
            ArtifactKind.AUDIO,
            f"yachiyo.{response_format if response_format == 'mp3' else 'ogg'}",
            mime,
            data=data,
        )

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with self.session.post(
                f"{self.base_url}/{path}",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                },
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    raise MediaError(f"media provider returned HTTP {response.status}")
                raw = await self._read_limited(response, 2 * 1024 * 1024)
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise MediaError("media provider request failed") from exc
        try:
            decoded = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MediaError("media provider returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise MediaError("media provider returned invalid JSON")
        return decoded

    async def _download(self, value: str, content_types: set[str]) -> bytes:
        current = self._artifact_url(value)
        try:
            for redirect in range(4):
                async with self.session.get(current, allow_redirects=False) as response:
                    if response.status in _REDIRECTS:
                        location = response.headers.get("Location")
                        if not location or redirect == 3:
                            raise MediaError("media redirect was rejected")
                        current = self._artifact_url(urljoin(current, location))
                        continue
                    if response.status != 200:
                        raise MediaError(f"artifact download returned HTTP {response.status}")
                    mime = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
                    if mime not in content_types:
                        raise MediaError("artifact content type is not allowed")
                    return await self._read_limited(response, self.settings.max_artifact_bytes)
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise MediaError("media artifact download failed") from exc
        raise MediaError("media redirect was rejected")

    @staticmethod
    def _artifact_url(value: str) -> str:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if (
            parsed.scheme != "https"
            or parsed.port not in {None, 443}
            or parsed.username
            or parsed.password
            or parsed.fragment
            or not _ARTIFACT_HOST.fullmatch(host)
        ):
            raise MediaError("provider returned an untrusted artifact URL")
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        if literal is not None and not literal.is_global:
            raise MediaError("provider returned a non-public artifact URL")
        return value

    @staticmethod
    def _text(value: str, limit: int) -> str:
        value = value.strip()
        if not value or len(value) > limit:
            raise MediaError(f"text must contain between 1 and {limit} characters")
        return value

    @staticmethod
    async def _read_limited(response: aiohttp.ClientResponse, limit: int) -> bytes:
        if response.content_length is not None and response.content_length > limit:
            raise MediaError("provider response exceeds the configured size")
        body = bytearray()
        async for chunk in response.content.iter_chunked(64 * 1024):
            body.extend(chunk)
            if len(body) > limit:
                raise MediaError("provider response exceeds the configured size")
        if not body:
            raise MediaError("provider returned an empty response")
        return bytes(body)

    async def close(self) -> None:
        await self.session.close()
