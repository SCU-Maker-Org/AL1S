"""Scoped knowledge index contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from al1s.domain import KnowledgeScope


@dataclass(frozen=True, slots=True)
class KnowledgeHit:
    entry_id: str
    content: str
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeIndex(Protocol):
    async def search(
        self,
        query: str,
        scope: KnowledgeScope,
        *,
        limit: int,
        include_memories: bool = True,
    ) -> tuple[KnowledgeHit, ...]: ...

    async def store_memory(
        self,
        entry_id: str,
        scope: KnowledgeScope,
        content: str,
        category: str,
        importance: float,
        *,
        consent: tuple[int, str] | None = None,
        effect_event_key: str | None = None,
    ) -> bool: ...

    async def delete_memory(
        self,
        entry_id: str,
        scope: KnowledgeScope,
        *,
        effect_event_key: str | None = None,
    ) -> bool: ...

    async def rebuild_vectors(self) -> int: ...

    async def stats(self) -> dict[str, Any]: ...
