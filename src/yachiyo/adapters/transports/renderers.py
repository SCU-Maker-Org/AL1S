"""Transport-safe plain-text splitting."""

from __future__ import annotations

import puremagic

_IMAGE_TYPES = {"image/gif", "image/jpeg", "image/png", "image/webp"}


def validated_mime(data: bytes, declared: str) -> str:
    if not declared.startswith("image/"):
        return declared
    try:
        detected = puremagic.from_string(data, mime=True)
    except puremagic.PureError as exc:
        raise ValueError("unsupported image attachment") from exc
    if detected not in _IMAGE_TYPES:
        raise ValueError("unsupported image attachment")
    return detected


def split_text(text: str, limit: int) -> tuple[str, ...]:
    if not text:
        return ()
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        boundary = max(
            remaining.rfind("\n", 0, limit + 1),
            remaining.rfind(" ", 0, limit + 1),
        )
        if boundary < limit // 3:
            boundary = limit
        chunks.append(remaining[:boundary].rstrip())
        remaining = remaining[boundary:].lstrip()
    if remaining:
        chunks.append(remaining)
    return tuple(chunks)
