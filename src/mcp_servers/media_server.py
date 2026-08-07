"""MCP tools that generate media artifacts for AL1S to send via Telegram."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import os
import re
import socket
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import aiohttp
from aiohttp.abc import AbstractResolver
from mcp.server.fastmcp import FastMCP

RECOMMENDED_IMAGE_SIZES = frozenset(
    {"2048x2048", "2688x1536", "1536x2688", "2368x1728", "1728x2368"}
)
IMAGE_SIZE_PATTERN = re.compile(r"([1-9][0-9]{1,4})[x*]([1-9][0-9]{1,4})")
MIN_IMAGE_PIXELS = 512 * 512
MAX_IMAGE_PIXELS = 2048 * 2048
SPEECH_MODEL_VOICES = {
    "qwen-audio-3.0-tts-plus": frozenset(
        {
            "longanlingxin",
            "longanlufeng",
        }
    ),
    "qwen-audio-3.0-tts-flash": frozenset(
        {
            "longanfengyue",
            "longanyuanfei",
            "longanlingxi",
            "longanxiaoxin",
            "longanhuan_v3.6",
            "longjielidou_v3.6",
            "longpaopao_v3.6",
            "longhuohuo_v3.6",
            "longchuanshu_v3.6",
            "loongmary",
            "loongeva_v3.6",
            "loongjohn",
        }
    ),
}
DEFAULT_SPEECH_MODEL = "qwen-audio-3.0-tts-plus"
DEFAULT_SPEECH_VOICE = "longanlingxin"
SPEECH_FORMATS = {
    "opus": (".ogg", "audio/ogg"),
    "mp3": (".mp3", "audio/mpeg"),
}

MAX_IMAGE_PROMPT_CHARS = 4_000
MAX_SPEECH_INPUT_CHARS = 4_096
MAX_IMAGE_BYTES = 10_000_000
MAX_SPEECH_BYTES = 50 * 1024 * 1024
MAX_PROVIDER_JSON_BYTES = 1024 * 1024
MAX_REMOTE_REDIRECTS = 3
DEFAULT_TTL_SECONDS = 24 * 60 * 60
MAX_TTL_SECONDS = 7 * 24 * 60 * 60
CAPTURE_NONCE_PATTERN = re.compile(r"[A-Za-z0-9_-]{32,128}")
OWNER_TAG_PATTERN = re.compile(r"[0-9a-f]{64}")
WORKSPACE_HOST_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9-]{1,126}\.(?:cn-beijing|ap-southeast-1)\.maas\.aliyuncs\.com$"
)
OSS_BUCKET_LABEL = r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]"
ARTIFACT_HOST_PATTERNS = (
    re.compile(
        rf"^{OSS_BUCKET_LABEL}\.oss-(?![a-z0-9-]*internal\.)"
        r"[a-z0-9-]+\.aliyuncs\.com$"
    ),
    re.compile(
        rf"^{OSS_BUCKET_LABEL}\.(?![a-z0-9-]*internal\.)"
        r"[a-z0-9-]+\.oss\.aliyuncs\.com$"
    ),
)
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
ALLOWED_PROVIDER_HOSTS = frozenset(
    {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"}
)

mcp = FastMCP(
    "AL1S Qwen Media Generator",
    instructions=(
        "Generate local media artifacts. Results are JSON metadata; the calling "
        "AL1S process is responsible for validating and sending each artifact."
    ),
)


def _environment(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _timeout_seconds() -> float:
    raw_timeout = _environment("AL1S_MEDIA_DASHSCOPE_TIMEOUT", default="180")
    try:
        timeout = float(raw_timeout or "180")
    except ValueError as exc:
        raise RuntimeError("invalid media provider timeout") from exc
    if not 1 <= timeout <= 600:
        raise RuntimeError("media provider timeout is outside the allowed range")
    return timeout


def _artifact_byte_limit(hard_limit: int) -> int:
    raw_limit = _environment("AL1S_MEDIA_MAX_ARTIFACT_BYTES", default=str(hard_limit))
    try:
        limit = int(raw_limit or str(hard_limit))
    except ValueError as exc:
        raise RuntimeError("invalid media artifact size limit") from exc
    if not 1024 <= limit <= MAX_SPEECH_BYTES:
        raise RuntimeError("media artifact size limit is outside the allowed range")
    return min(limit, hard_limit)


def _validate_provider_base_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError("invalid DashScope base URL") from exc
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or port not in (None, 443):
        raise RuntimeError("DashScope base URL must use HTTPS on the default port")
    if parsed.username is not None or parsed.password is not None:
        raise RuntimeError("credentials are forbidden in the DashScope base URL")
    if host not in ALLOWED_PROVIDER_HOSTS and not WORKSPACE_HOST_PATTERN.fullmatch(
        host
    ):
        raise RuntimeError("DashScope base URL host is not an official endpoint")
    path = parsed.path.rstrip("/")
    if path != "/api/v1" or parsed.query or parsed.fragment:
        raise RuntimeError("DashScope base URL must end with /api/v1")
    return urlunsplit(("https", parsed.netloc, path, "", ""))


def _provider_settings() -> tuple[str, str, float]:
    api_key = _environment("AL1S_MEDIA_DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("media provider credentials are not configured")
    base_url = _environment(
        "AL1S_MEDIA_DASHSCOPE_BASE_URL",
        default="https://dashscope.aliyuncs.com/api/v1",
    )
    return api_key, _validate_provider_base_url(base_url or ""), _timeout_seconds()


def _ensure_public_addresses(addresses: Iterable[str], host: str) -> None:
    found = False
    for raw_address in addresses:
        found = True
        try:
            address = ipaddress.ip_address(raw_address)
        except ValueError as exc:
            raise RuntimeError("media endpoint DNS response is invalid") from exc
        if not address.is_global:
            raise RuntimeError("media endpoint resolved to a non-public address")
    if not found:
        raise RuntimeError(f"media endpoint DNS returned no addresses for {host}")


class _PublicOnlyResolver(AbstractResolver):
    """Validate the addresses actually handed to aiohttp's connector."""

    def __init__(self) -> None:
        self._delegate = aiohttp.DefaultResolver()

    async def resolve(
        self, host: str, port: int = 0, family: int = socket.AF_INET
    ) -> list[dict[str, Any]]:
        records = await self._delegate.resolve(host, port, family)
        _ensure_public_addresses((str(record["host"]) for record in records), host)
        return records

    async def close(self) -> None:
        await self._delegate.close()


def _get_header(headers: Mapping[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if str(key).lower() == name.lower():
            return str(value)
    return None


async def _read_limited_body(response: Any, max_bytes: int) -> bytes:
    content_length = _get_header(response.headers, "Content-Length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError as exc:
            raise RuntimeError(
                "media provider returned invalid content length"
            ) from exc
        if declared_size < 0 or declared_size > max_bytes:
            raise RuntimeError("media provider response exceeds the size limit")

    body = bytearray()
    async for chunk in response.content.iter_chunked(64 * 1024):
        body.extend(chunk)
        if len(body) > max_bytes:
            raise RuntimeError("media provider response exceeds the size limit")
    if not body:
        raise RuntimeError("media provider returned an empty response")
    return bytes(body)


async def _post_provider_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    api_key, base_url, timeout = _provider_settings()
    url = f"{base_url}/{path.lstrip('/')}"
    connector = aiohttp.TCPConnector(resolver=_PublicOnlyResolver(), ttl_dns_cache=0)
    try:
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as session:
            async with session.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    raise RuntimeError("media provider request failed")
                content_type = (
                    (_get_header(response.headers, "Content-Type") or "")
                    .split(";", 1)[0]
                    .strip()
                    .lower()
                )
                if content_type not in {"application/json", "text/json"}:
                    raise RuntimeError("media provider returned invalid JSON metadata")
                raw_body = await _read_limited_body(response, MAX_PROVIDER_JSON_BYTES)
    except RuntimeError:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError):
        raise RuntimeError("media provider request failed") from None

    try:
        decoded = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("media provider returned invalid JSON metadata") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("media provider returned invalid JSON metadata")
    return decoded


def _normalize_artifact_url(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 8192:
        raise RuntimeError("media provider returned an invalid artifact URL")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError("media provider returned an invalid artifact URL") from exc
    host = (parsed.hostname or "").lower()
    if parsed.username is not None or parsed.password is not None:
        raise RuntimeError("media artifact URL contains credentials")
    if parsed.scheme == "http" and host.endswith(".aliyuncs.com") and port is None:
        parsed = parsed._replace(scheme="https")
        port = None
    if parsed.scheme != "https" or port not in (None, 443):
        raise RuntimeError("media artifact URL must use HTTPS")
    if not host or host.endswith(".") or not parsed.path.startswith("/"):
        raise RuntimeError("media provider returned an invalid artifact URL")
    if not any(pattern.fullmatch(host) for pattern in ARTIFACT_HOST_PATTERNS):
        raise RuntimeError(
            "media artifact URL host is not an approved public OSS bucket"
        )
    try:
        literal_address = ipaddress.ip_address(host)
    except ValueError:
        literal_address = None
    if literal_address is not None and not literal_address.is_global:
        raise RuntimeError("media artifact URL targets a non-public address")
    if parsed.fragment:
        raise RuntimeError("media artifact URL must not contain a fragment")
    return urlunsplit(parsed)


def _validate_artifact_signature(content: bytes, media_format: str) -> None:
    valid = False
    if media_format == "png":
        valid = content.startswith(b"\x89PNG\r\n\x1a\n")
    elif media_format == "opus":
        valid = content.startswith(b"OggS") and b"OpusHead" in content[:256]
    elif media_format == "mp3":
        valid = content.startswith(b"ID3") or (
            len(content) >= 2 and content[0] == 0xFF and content[1] & 0xE0 == 0xE0
        )
    if not valid:
        raise RuntimeError("media provider returned an invalid artifact format")


async def _download_remote_artifact(
    value: str,
    *,
    media_format: str,
    allowed_content_types: frozenset[str],
    max_bytes: int,
) -> bytes:
    current_url = _normalize_artifact_url(value)
    connector = aiohttp.TCPConnector(resolver=_PublicOnlyResolver(), ttl_dns_cache=0)
    try:
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=_timeout_seconds()),
        ) as session:
            for redirect_count in range(MAX_REMOTE_REDIRECTS + 1):
                async with session.get(
                    current_url,
                    headers={"Accept": "*/*"},
                    allow_redirects=False,
                ) as response:
                    if response.status in REDIRECT_STATUSES:
                        location = _get_header(response.headers, "Location")
                        if not location or redirect_count >= MAX_REMOTE_REDIRECTS:
                            raise RuntimeError("media artifact redirect was rejected")
                        current_url = _normalize_artifact_url(
                            urljoin(current_url, location)
                        )
                        continue
                    if response.status != 200:
                        raise RuntimeError("media artifact download failed")
                    content_type = (
                        (_get_header(response.headers, "Content-Type") or "")
                        .split(";", 1)[0]
                        .strip()
                        .lower()
                    )
                    if content_type not in allowed_content_types:
                        raise RuntimeError(
                            "media artifact has an unsupported content type"
                        )
                    content = await _read_limited_body(response, max_bytes)
                    _validate_artifact_signature(content, media_format)
                    return content
    except RuntimeError:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError):
        raise RuntimeError("media artifact download failed") from None
    raise RuntimeError("media artifact redirect was rejected")


def _output_root() -> Path:
    configured = os.getenv("AL1S_MEDIA_OUTPUT_DIR")
    if not configured:
        raise RuntimeError("AL1S_MEDIA_OUTPUT_DIR is not configured")
    root = Path(configured).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not root.is_dir():
        raise RuntimeError("media output location is not a directory")
    return root


def _ttl_seconds() -> int:
    raw_value = os.getenv("AL1S_MEDIA_TTL_SECONDS", str(DEFAULT_TTL_SECONDS))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError("invalid media artifact TTL") from exc
    if not 60 <= value <= MAX_TTL_SECONDS:
        raise RuntimeError("media artifact TTL is outside the allowed range")
    return value


def _validate_text(value: str, *, field: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field} must not be empty")
    if len(cleaned) > max_chars:
        raise ValueError(f"{field} exceeds {max_chars} characters")
    return cleaned


def _validate_enum(value: str, *, field: str, allowed: Any) -> str:
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"invalid {field}; expected one of: {choices}")
    return value


def _validate_image_size(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("size must be a string")
    match = IMAGE_SIZE_PATTERN.fullmatch(value.strip().lower())
    if match is None:
        raise ValueError("invalid size; expected WIDTHxHEIGHT")
    width, height = (int(part) for part in match.groups())
    pixels = width * height
    if not MIN_IMAGE_PIXELS <= pixels <= MAX_IMAGE_PIXELS:
        raise ValueError(
            "invalid size; total pixels must be between 512x512 and 2048x2048"
        )
    return f"{width}x{height}"


def _validate_capture_binding(capture_nonce: str, owner_tag: str) -> tuple[str, str]:
    """Validate opaque binding values injected by the trusted AL1S MCP client."""
    if not isinstance(capture_nonce, str) or not CAPTURE_NONCE_PATTERN.fullmatch(
        capture_nonce
    ):
        raise ValueError("invalid media capture nonce")
    if not isinstance(owner_tag, str) or not OWNER_TAG_PATTERN.fullmatch(owner_tag):
        raise ValueError("invalid media capture owner")
    return capture_nonce, owner_tag


async def _request_image(
    *,
    prompt: str,
    size: str,
    max_bytes: int,
) -> bytes:
    response = await _post_provider_json(
        "services/aigc/multimodal-generation/generation",
        {
            "model": _environment(
                "AL1S_MEDIA_IMAGE_MODEL", default="qwen-image-2.0-pro"
            ),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
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
        content_items = response["output"]["choices"][0]["message"]["content"]
        artifact_url = next(
            item["image"]
            for item in content_items
            if isinstance(item, dict) and isinstance(item.get("image"), str)
        )
    except (KeyError, IndexError, TypeError, StopIteration) as exc:
        raise RuntimeError("image provider returned no artifact") from exc
    return await _download_remote_artifact(
        artifact_url,
        media_format="png",
        allowed_content_types=frozenset(
            {"image/png", "application/octet-stream", "binary/octet-stream"}
        ),
        max_bytes=max_bytes,
    )


async def _request_speech(
    *,
    text: str,
    voice: str,
    response_format: str,
    speed: float,
    max_bytes: int,
    model: str,
) -> bytes:
    response = await _post_provider_json(
        "services/audio/tts/SpeechSynthesizer",
        {
            "model": model,
            "input": {
                "text": text,
                "voice": voice,
                "format": response_format,
                "sample_rate": 24_000,
                "rate": speed,
                "enable_aigc_tag": True,
            },
        },
    )
    try:
        artifact_url = response["output"]["audio"]["url"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("speech provider returned no artifact") from exc
    if not isinstance(artifact_url, str):
        raise RuntimeError("speech provider returned no artifact")

    allowed_types = {
        "opus": frozenset(
            {
                "audio/ogg",
                "audio/opus",
                "application/ogg",
                "application/octet-stream",
                "binary/octet-stream",
            }
        ),
        "mp3": frozenset(
            {
                "audio/mpeg",
                "audio/mp3",
                "application/octet-stream",
                "binary/octet-stream",
            }
        ),
    }
    return await _download_remote_artifact(
        artifact_url,
        media_format=response_format,
        allowed_content_types=allowed_types[response_format],
        max_bytes=max_bytes,
    )


def _write_artifact(
    content: bytes,
    *,
    kind: str,
    suffix: str,
    mime_type: str,
    max_bytes: int,
    capture_nonce: str,
    owner_tag: str,
) -> dict[str, Any]:
    if not content:
        raise RuntimeError("media provider returned an empty artifact")
    if len(content) > max_bytes:
        raise RuntimeError("generated media exceeds the configured size limit")

    root = _output_root()
    ttl_seconds = _ttl_seconds()
    capture_nonce, owner_tag = _validate_capture_binding(capture_nonce, owner_tag)
    destination_dir = root / "requests" / owner_tag / capture_nonce / kind
    destination_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = destination_dir / f"{uuid.uuid4().hex}{suffix}"

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp-", suffix=suffix, dir=destination_dir
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "wb") as file_handle:
            file_handle.write(content)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.close(file_descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise

    relative_path = destination.relative_to(root).as_posix()
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
    return {
        "artifact_id": destination.stem,
        "relative_path": relative_path,
        "kind": kind,
        "mime_type": mime_type,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
        "capture_nonce": capture_nonce,
        "owner_tag": owner_tag,
    }


def _artifact_json(artifact: dict[str, Any]) -> str:
    return json.dumps(artifact, ensure_ascii=False, separators=(",", ":"))


def _prune_expired_artifacts(root: Path, ttl_seconds: int) -> None:
    """Best-effort cleanup of regular artifacts left beyond the configured TTL."""
    cutoff = datetime.now(timezone.utc).timestamp() - ttl_seconds
    requests_root = root / "requests"
    if not requests_root.is_dir() or requests_root.is_symlink():
        return
    for current_dir, directory_names, file_names in os.walk(
        requests_root, topdown=False, followlinks=False
    ):
        current = Path(current_dir)
        for file_name in file_names:
            candidate = current / file_name
            try:
                if (
                    candidate.is_file()
                    and not candidate.is_symlink()
                    and candidate.stat().st_mtime < cutoff
                ):
                    candidate.unlink()
            except OSError:
                continue
        for directory_name in directory_names:
            directory = current / directory_name
            try:
                if not directory.is_symlink():
                    directory.rmdir()
            except OSError:
                continue


@mcp.tool(structured_output=False)
async def generate_image(
    prompt: str,
    size: str = "2048x2048",
    *,
    al1s_capture_nonce: str,
    al1s_capture_owner: str,
) -> str:
    """Generate one PNG with Qwen-Image and return its local artifact metadata.

    Size accepts WIDTHxHEIGHT or WIDTH*HEIGHT. For Qwen-Image 2.0, total pixels
    must be between 512x512 and 2048x2048; recommended sizes include
    2048x2048, 2688x1536, 1536x2688, 2368x1728, and 1728x2368.
    """
    prompt = _validate_text(prompt, field="prompt", max_chars=MAX_IMAGE_PROMPT_CHARS)
    size = _validate_image_size(size)
    al1s_capture_nonce, al1s_capture_owner = _validate_capture_binding(
        al1s_capture_nonce, al1s_capture_owner
    )
    _prune_expired_artifacts(_output_root(), _ttl_seconds())
    max_bytes = _artifact_byte_limit(MAX_IMAGE_BYTES)

    try:
        content = await _request_image(
            prompt=prompt,
            size=size,
            max_bytes=max_bytes,
        )
    except Exception:
        raise RuntimeError("image generation failed") from None

    return _artifact_json(
        _write_artifact(
            content,
            kind="image",
            suffix=".png",
            mime_type="image/png",
            max_bytes=max_bytes,
            capture_nonce=al1s_capture_nonce,
            owner_tag=al1s_capture_owner,
        )
    )


@mcp.tool(structured_output=False)
async def synthesize_speech(
    text: str,
    voice: str = DEFAULT_SPEECH_VOICE,
    response_format: str = "opus",
    speed: float = 1.0,
    *,
    al1s_capture_nonce: str,
    al1s_capture_owner: str,
) -> str:
    """Synthesize Qwen-Audio speech and return its local artifact metadata."""
    text = _validate_text(text, field="text", max_chars=MAX_SPEECH_INPUT_CHARS)
    model = _environment("AL1S_MEDIA_TTS_MODEL", default=DEFAULT_SPEECH_MODEL)
    if model not in SPEECH_MODEL_VOICES:
        raise ValueError("configured speech model is not a supported Qwen-Audio model")
    if voice == DEFAULT_SPEECH_VOICE:
        voice = _environment("AL1S_MEDIA_TTS_VOICE", default=voice) or voice
    voice = _validate_enum(
        voice, field=f"voice for {model}", allowed=SPEECH_MODEL_VOICES[model]
    )
    response_format = _validate_enum(
        response_format, field="response_format", allowed=SPEECH_FORMATS
    )
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        raise ValueError("speed must be a number")
    speed = float(speed)
    if not 0.5 <= speed <= 2.0:
        raise ValueError("speed must be between 0.5 and 2.0")
    al1s_capture_nonce, al1s_capture_owner = _validate_capture_binding(
        al1s_capture_nonce, al1s_capture_owner
    )
    _prune_expired_artifacts(_output_root(), _ttl_seconds())
    max_bytes = _artifact_byte_limit(MAX_SPEECH_BYTES)

    try:
        content = await _request_speech(
            text=text,
            voice=voice,
            response_format=response_format,
            speed=speed,
            max_bytes=max_bytes,
            model=model,
        )
    except Exception:
        raise RuntimeError("speech synthesis failed") from None

    suffix, mime_type = SPEECH_FORMATS[response_format]
    return _artifact_json(
        _write_artifact(
            content,
            kind="voice",
            suffix=suffix,
            mime_type=mime_type,
            max_bytes=max_bytes,
            capture_nonce=al1s_capture_nonce,
            owner_tag=al1s_capture_owner,
        )
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
