"""Reverse image search contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ImageSearchHit:
    engine: str
    title: str
    url: str
    similarity: float | None = None
    preview_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ReverseImageSearch(Protocol):
    async def search_bytes(self, data: bytes, mime_type: str) -> tuple[ImageSearchHit, ...]: ...

    async def search_url(self, url: str) -> tuple[ImageSearchHit, ...]: ...
