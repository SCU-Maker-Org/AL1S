"""Deterministic heading and paragraph based text chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True, slots=True)
class TextChunk:
    ordinal: int
    heading: str
    content: str
    content_hash: str


def _sections(text: str) -> list[tuple[str, str]]:
    heading = ""
    paragraphs: list[str] = []
    result: list[tuple[str, str]] = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        cleaned = block.strip()
        if not cleaned:
            continue
        match = _HEADING.match(cleaned.splitlines()[0])
        if match:
            if paragraphs:
                result.append((heading, "\n\n".join(paragraphs)))
                paragraphs = []
            heading = match.group(2).strip()
            remainder = "\n".join(cleaned.splitlines()[1:]).strip()
            if remainder:
                paragraphs.append(remainder)
        else:
            paragraphs.append(cleaned)
    if paragraphs:
        result.append((heading, "\n\n".join(paragraphs)))
    return result


def chunk_text(
    text: str,
    *,
    max_chars: int,
    overlap_chars: int,
    max_chunks: int | None = None,
) -> tuple[TextChunk, ...]:
    chunks: list[TextChunk] = []
    for heading, section in _sections(text):
        start = 0
        while start < len(section):
            maximum_end = min(start + max_chars, len(section))
            end = maximum_end
            if maximum_end < len(section):
                minimum_end = start + max_chars // 2
                boundaries = (
                    section.rfind("\n", minimum_end, maximum_end),
                    section.rfind(" ", minimum_end, maximum_end),
                    section.rfind("。", minimum_end, maximum_end),
                    section.rfind(".", minimum_end, maximum_end),
                )
                boundary = max(boundaries)
                if boundary >= minimum_end:
                    end = boundary + 1
            content = section[start:end].strip()
            if content:
                digest = hashlib.sha256(content.encode()).hexdigest()
                chunks.append(TextChunk(len(chunks), heading, content, digest))
                if max_chunks is not None and len(chunks) > max_chunks:
                    raise ValueError("document produces too many chunks")
            if end >= len(section):
                break
            start = max(start + 1, end - overlap_chars)
    return tuple(chunks)
