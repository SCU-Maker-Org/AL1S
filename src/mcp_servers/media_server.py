"""MCP tools that generate media artifacts for AL1S to send via Telegram."""

from __future__ import annotations

import base64
import binascii
import hashlib
import inspect
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI

IMAGE_SIZES = frozenset({"1024x1024", "1536x1024", "1024x1536"})
IMAGE_QUALITIES = frozenset({"low", "medium", "high", "auto"})
IMAGE_FORMATS = {
    "png": (".png", "image/png"),
    "jpeg": (".jpg", "image/jpeg"),
    "webp": (".webp", "image/webp"),
}
SPEECH_VOICES = frozenset(
    {
        "alloy",
        "ash",
        "ballad",
        "cedar",
        "coral",
        "echo",
        "fable",
        "marin",
        "nova",
        "onyx",
        "sage",
        "shimmer",
        "verse",
    }
)
SPEECH_FORMATS = {
    "opus": (".ogg", "audio/ogg"),
    "mp3": (".mp3", "audio/mpeg"),
}

MAX_IMAGE_PROMPT_CHARS = 4_000
MAX_SPEECH_INPUT_CHARS = 4_096
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_SPEECH_BYTES = 50 * 1024 * 1024
DEFAULT_TTL_SECONDS = 24 * 60 * 60
MAX_TTL_SECONDS = 7 * 24 * 60 * 60
CAPTURE_NONCE_PATTERN = re.compile(r"[A-Za-z0-9_-]{32,128}")
OWNER_TAG_PATTERN = re.compile(r"[0-9a-f]{64}")

mcp = FastMCP(
    "AL1S Media Generator",
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


def _create_openai_client() -> AsyncOpenAI:
    """Create the provider client; kept separate so tests never need the network."""
    api_key = _environment("AL1S_MEDIA_OPENAI_API_KEY", "OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("media provider credentials are not configured")

    kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 2}
    base_url = _environment("AL1S_MEDIA_OPENAI_BASE_URL", "OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    timeout = _environment("AL1S_MEDIA_OPENAI_TIMEOUT", default="180")
    try:
        kwargs["timeout"] = float(timeout or "180")
    except ValueError as exc:
        raise RuntimeError("invalid media provider timeout") from exc
    return AsyncOpenAI(**kwargs)


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


def _validate_capture_binding(capture_nonce: str, owner_tag: str) -> tuple[str, str]:
    """Validate opaque binding values injected by the trusted AL1S MCP client."""
    if not isinstance(capture_nonce, str) or not CAPTURE_NONCE_PATTERN.fullmatch(
        capture_nonce
    ):
        raise ValueError("invalid media capture nonce")
    if not isinstance(owner_tag, str) or not OWNER_TAG_PATTERN.fullmatch(owner_tag):
        raise ValueError("invalid media capture owner")
    return capture_nonce, owner_tag


def _read_image_response(response: Any) -> bytes:
    data = getattr(response, "data", None)
    encoded = getattr(data[0], "b64_json", None) if data else None
    if not encoded or not isinstance(encoded, str):
        raise RuntimeError("image provider returned no inline image data")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError("image provider returned invalid image data") from exc
    return content


async def _read_speech_response(response: Any) -> bytes:
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content

    read = getattr(response, "read", None)
    if not callable(read):
        raise RuntimeError("speech provider returned no audio data")
    content = read()
    if inspect.isawaitable(content):
        content = await content
    if not isinstance(content, bytes):
        raise RuntimeError("speech provider returned invalid audio data")
    return content


async def _request_image(
    client: AsyncOpenAI,
    *,
    prompt: str,
    size: str,
    quality: str,
    output_format: str,
) -> bytes:
    response = await client.images.generate(
        model=_environment("AL1S_MEDIA_IMAGE_MODEL", default="gpt-image-2"),
        prompt=prompt,
        size=size,
        quality=quality,
        output_format=output_format,
        response_format="b64_json",
        n=1,
    )
    return _read_image_response(response)


async def _request_speech(
    client: AsyncOpenAI,
    *,
    text: str,
    voice: str,
    response_format: str,
    speed: float,
) -> bytes:
    response = await client.audio.speech.create(
        model=_environment("AL1S_MEDIA_TTS_MODEL", default="tts-1"),
        input=text,
        voice=voice,
        response_format=response_format,
        speed=speed,
    )
    return await _read_speech_response(response)


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
    size: str = "1024x1024",
    quality: str = "medium",
    output_format: str = "png",
    *,
    al1s_capture_nonce: str,
    al1s_capture_owner: str,
) -> str:
    """Generate one image and return JSON metadata for its local artifact."""
    prompt = _validate_text(prompt, field="prompt", max_chars=MAX_IMAGE_PROMPT_CHARS)
    size = _validate_enum(size, field="size", allowed=IMAGE_SIZES)
    quality = _validate_enum(quality, field="quality", allowed=IMAGE_QUALITIES)
    output_format = _validate_enum(
        output_format, field="output_format", allowed=IMAGE_FORMATS
    )
    al1s_capture_nonce, al1s_capture_owner = _validate_capture_binding(
        al1s_capture_nonce, al1s_capture_owner
    )
    _prune_expired_artifacts(_output_root(), _ttl_seconds())

    try:
        content = await _request_image(
            _create_openai_client(),
            prompt=prompt,
            size=size,
            quality=quality,
            output_format=output_format,
        )
    except Exception:
        raise RuntimeError("image generation failed") from None

    suffix, mime_type = IMAGE_FORMATS[output_format]
    return _artifact_json(
        _write_artifact(
            content,
            kind="image",
            suffix=suffix,
            mime_type=mime_type,
            max_bytes=MAX_IMAGE_BYTES,
            capture_nonce=al1s_capture_nonce,
            owner_tag=al1s_capture_owner,
        )
    )


@mcp.tool(structured_output=False)
async def synthesize_speech(
    text: str,
    voice: str = "alloy",
    response_format: str = "opus",
    speed: float = 1.0,
    *,
    al1s_capture_nonce: str,
    al1s_capture_owner: str,
) -> str:
    """Synthesize speech and return JSON metadata for its local artifact."""
    text = _validate_text(text, field="text", max_chars=MAX_SPEECH_INPUT_CHARS)
    if voice == "alloy":
        voice = _environment("AL1S_MEDIA_TTS_VOICE", default=voice) or voice
    voice = _validate_enum(voice, field="voice", allowed=SPEECH_VOICES)
    response_format = _validate_enum(
        response_format, field="response_format", allowed=SPEECH_FORMATS
    )
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        raise ValueError("speed must be a number")
    speed = float(speed)
    if not 0.25 <= speed <= 4.0:
        raise ValueError("speed must be between 0.25 and 4.0")
    al1s_capture_nonce, al1s_capture_owner = _validate_capture_binding(
        al1s_capture_nonce, al1s_capture_owner
    )
    _prune_expired_artifacts(_output_root(), _ttl_seconds())

    try:
        content = await _request_speech(
            _create_openai_client(),
            text=text,
            voice=voice,
            response_format=response_format,
            speed=speed,
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
            max_bytes=MAX_SPEECH_BYTES,
            capture_nonce=al1s_capture_nonce,
            owner_tag=al1s_capture_owner,
        )
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
