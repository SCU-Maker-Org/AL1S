from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterator

from .documents import Chunk, Document, stable_sha256

_MARKDOWN_HEADING = re.compile(
    r"^[ ]{0,3}(?P<marks>#{1,6})[ \t]+(?P<title>.+?)[ \t]*#*[ \t]*$"
)
_MARKDOWN_FENCE = re.compile(r"^[ ]{0,3}(?P<fence>`{3,}|~{3,})(?P<rest>.*)$")
_RST_HEADING = re.compile(
    r"(?m)^(?P<title>[^\n]+)\n(?P<underline>[=\-~^\"'+*#`:_<>]{3,})[ \t]*$"
)
_RST_LEVELS = "=-~^\"'+*#`:_<>"


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    max_chars: int = 1800
    overlap_chars: int = 240

    def __post_init__(self) -> None:
        if self.max_chars <= 0:
            raise ValueError("max_chars must be greater than zero")
        if self.overlap_chars < 0:
            raise ValueError("overlap_chars must not be negative")
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")


@dataclass(frozen=True, slots=True)
class _Section:
    start: int
    end: int
    heading_path: tuple[str, ...]


def chunk_document(
    document: Document, config: ChunkingConfig | None = None
) -> tuple[Chunk, ...]:
    settings = config or ChunkingConfig()
    chunks: list[Chunk] = []
    for section in _sections(document):
        section_text = document.content[section.start : section.end]
        for start, end in _split_text(section_text, settings):
            raw = section_text[start:end]
            left_trim = len(raw) - len(raw.lstrip())
            right_trim = len(raw) - len(raw.rstrip())
            absolute_start = section.start + start + left_trim
            absolute_end = section.start + end - right_trim
            content = document.content[absolute_start:absolute_end]
            if not content:
                continue
            ordinal = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=stable_sha256("chunk", document.document_id, str(ordinal)),
                    document_id=document.document_id,
                    source_path=document.source_path,
                    ordinal=ordinal,
                    content=content,
                    content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    start_char=absolute_start,
                    end_char=absolute_end,
                    heading_path=section.heading_path,
                    domains=document.domains,
                    metadata=document.metadata,
                )
            )
    return tuple(chunks)


def _sections(document: Document) -> tuple[_Section, ...]:
    if document.format not in {"markdown", "rst", "html"}:
        return (_Section(0, len(document.content), ()),)

    matches: list[tuple[int, int, str]] = []
    if document.format in {"markdown", "html"}:
        matches.extend(_markdown_headings(document.content))
    if document.format == "rst":
        for match in _RST_HEADING.finditer(document.content):
            underline = match.group("underline")
            if len(set(underline)) != 1:
                continue
            level = _RST_LEVELS.find(underline[0]) + 1
            matches.append((match.start(), max(level, 1), match.group("title")))
    matches.sort(key=lambda item: item[0])
    if not matches:
        return (_Section(0, len(document.content), ()),)

    sections: list[_Section] = []
    if matches[0][0] > 0 and document.content[: matches[0][0]].strip():
        sections.append(_Section(0, matches[0][0], ()))
    path: list[str] = []
    for index, (start, level, title) in enumerate(matches):
        path = path[: level - 1]
        path.append(title.strip())
        end = (
            matches[index + 1][0] if index + 1 < len(matches) else len(document.content)
        )
        sections.append(_Section(start, end, tuple(path)))
    return tuple(sections)


def _markdown_headings(text: str) -> list[tuple[int, int, str]]:
    headings: list[tuple[int, int, str]] = []
    active_fence: tuple[str, int] | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        fence_match = _MARKDOWN_FENCE.match(content)
        if fence_match:
            marker = fence_match.group("fence")
            marker_type = marker[0]
            if active_fence is None:
                active_fence = (marker_type, len(marker))
            elif (
                marker_type == active_fence[0]
                and len(marker) >= active_fence[1]
                and not fence_match.group("rest").strip()
            ):
                active_fence = None
            offset += len(line)
            continue
        if active_fence is None:
            heading_match = _MARKDOWN_HEADING.match(content)
            if heading_match:
                headings.append(
                    (
                        offset + heading_match.start(),
                        len(heading_match.group("marks")),
                        heading_match.group("title"),
                    )
                )
        offset += len(line)
    return headings


def _split_text(text: str, config: ChunkingConfig) -> Iterator[tuple[int, int]]:
    start = 0
    text_length = len(text)
    while start < text_length:
        hard_end = min(start + config.max_chars, text_length)
        end = hard_end
        if hard_end < text_length:
            lower_bound = start + max(config.max_chars // 2, config.overlap_chars + 1)
            end = _preferred_boundary(text, lower_bound, hard_end)
            if end <= start + config.overlap_chars:
                end = hard_end
        if text[start:end].strip():
            yield start, end
        if end >= text_length:
            break
        start = max(start + 1, end - config.overlap_chars)


def _preferred_boundary(text: str, lower_bound: int, hard_end: int) -> int:
    for separator in ("\n\n", "\n", ". ", "。", "! ", "? ", " "):
        position = text.rfind(separator, lower_bound, hard_end)
        if position >= 0:
            return position + len(separator)
    return hard_end
