"""Scoped SQLite full-text search for documents and memories."""

from __future__ import annotations

import asyncio
import hashlib
import os
import stat
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from itertools import zip_longest
from pathlib import Path
from typing import Any

from yachiyo.adapters.storage import Database
from yachiyo.config import AppSettings
from yachiyo.domain import KnowledgeScope
from yachiyo.domain.search_text import fts_document, search_terms
from yachiyo.domain.versions import INGESTION_PIPELINE_VERSION, LEXICAL_PIPELINE_VERSION
from yachiyo.ports.knowledge import KnowledgeHit

from .chunking import TextChunk, chunk_text
from .html import html_to_text

_ALLOWED_SUFFIXES = {".txt", ".md", ".markdown", ".html", ".htm"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _fts_query(text: str) -> str:
    tokens = search_terms(text, limit=32)
    safe = [f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens]
    return " OR ".join(safe)


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    path: Path
    source_root: str
    relative_path: str
    document_id: str
    title: str
    content: str
    content_hash: str
    revision: str
    chunks: tuple[TextChunk, ...]


@dataclass(frozen=True, slots=True)
class IngestionReport:
    discovered: int = 0
    indexed: int = 0
    unchanged: int = 0
    skipped: int = 0
    deleted: int = 0


class SQLiteKnowledgeIndex:
    def __init__(self, database: Database, settings: AppSettings):
        self.database = database
        self.settings = settings

    async def search(
        self,
        query: str,
        scope: KnowledgeScope,
        *,
        limit: int,
        include_memories: bool = True,
    ) -> tuple[KnowledgeHit, ...]:
        expression = _fts_query(query)
        if not expression or limit <= 0:
            return ()
        scopes = (scope,) if scope.key == "public" else (scope, KnowledgeScope("public"))
        rankings: list[list[KnowledgeHit]] = []
        include_memories = (
            include_memories
            and self.settings.memory.enabled
            and self.settings.memory.prompt_results > 0
        )
        for allowed_scope in scopes:
            rankings.extend(
                await self._search_scope(
                    expression, allowed_scope, include_memories=include_memories
                )
            )
        result: list[KnowledgeHit] = []
        memory_count = 0
        # Interleave the BM25 rankings for each source and scope.
        for rank, hits in enumerate(zip_longest(*rankings), start=1):
            for hit in hits:
                if hit is None:
                    continue
                if hit.source == "memory":
                    if memory_count >= self.settings.memory.prompt_results:
                        continue
                    memory_count += 1
                result.append(replace(hit, score=1.0 / (60 + rank)))
                if len(result) == limit:
                    return tuple(result)
        return tuple(result)

    async def _search_scope(
        self,
        expression: str,
        scope: KnowledgeScope,
        *,
        include_memories: bool,
    ) -> tuple[list[KnowledgeHit], ...]:
        document_hits: list[KnowledgeHit] = []
        memory_hits: list[KnowledgeHit] = []
        async with self.database.operation():
            connection = self.database.connection
            cursor = await connection.execute(
                """
                SELECT c.id, c.content, c.heading, d.source_uri, bm25(chunks_fts) AS rank
                FROM chunks_fts f
                JOIN chunks c ON c.id=CAST(f.chunk_id AS INTEGER) AND c.scope=f.scope
                JOIN documents d ON d.id=c.document_id AND d.scope=c.scope
                WHERE chunks_fts MATCH ? AND f.scope=?
                ORDER BY rank LIMIT ?
                """,
                (expression, scope.key, self.settings.rag.sparse_candidates),
            )
            for row in await cursor.fetchall():
                document_hits.append(
                    KnowledgeHit(
                        entry_id=str(row["id"]),
                        content=row["content"],
                        score=-float(row["rank"]),
                        source="document",
                        metadata={
                            "heading": row["heading"],
                            "source_uri": row["source_uri"],
                        },
                    )
                )
            if include_memories:
                cursor = await connection.execute(
                    """
                    SELECT m.id, m.content, m.category, bm25(memories_fts) AS rank
                    FROM memories_fts f
                    JOIN memories m ON m.id=f.memory_id AND m.scope=f.scope
                    WHERE memories_fts MATCH ? AND f.scope=?
                    ORDER BY rank LIMIT ?
                    """,
                    (expression, scope.key, self.settings.memory.prompt_results),
                )
                for row in await cursor.fetchall():
                    memory_hits.append(
                        KnowledgeHit(
                            entry_id=row["id"],
                            content=row["content"],
                            score=-float(row["rank"]),
                            source="memory",
                            metadata={"category": row["category"]},
                        )
                    )
        return document_hits, memory_hits

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
    ) -> bool:
        async with self.database.operation():
            connection = self.database.connection
            await connection.execute("BEGIN IMMEDIATE")
            try:
                if effect_event_key is not None:
                    cursor = await connection.execute(
                        "SELECT 1 FROM command_effects WHERE event_key=?",
                        (effect_event_key,),
                    )
                    if await cursor.fetchone() is not None:
                        await connection.rollback()
                        return True
                if consent is not None:
                    cursor = await connection.execute(
                        """
                        SELECT 1 FROM group_settings
                        WHERE chat_id=? AND memory_enabled=1 AND consent_at=?
                        """,
                        consent,
                    )
                    if await cursor.fetchone() is None:
                        await connection.rollback()
                        return False
                cursor = await connection.execute(
                    "SELECT scope FROM memories WHERE id=?", (entry_id,)
                )
                existing = await cursor.fetchone()
                if existing is not None and existing["scope"] != scope.key:
                    raise ValueError("memory ID belongs to a different scope")
                now = _now()
                await connection.execute(
                    """
                    INSERT INTO memories(
                        id, scope, content, category, importance, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET content=excluded.content,
                        category=excluded.category, importance=excluded.importance,
                        updated_at=excluded.updated_at
                    """,
                    (entry_id, scope.key, content, category, importance, now, now),
                )
                await connection.execute(
                    "DELETE FROM memories_fts WHERE memory_id=? AND scope=?",
                    (entry_id, scope.key),
                )
                await connection.execute(
                    "INSERT INTO memories_fts(memory_id, scope, content) VALUES (?, ?, ?)",
                    (entry_id, scope.key, fts_document(content)),
                )
                if effect_event_key is not None:
                    await connection.execute(
                        """
                        INSERT INTO command_effects(event_key, kind, created_at)
                        VALUES (?, 'memory_store', ?)
                        """,
                        (effect_event_key, _now()),
                    )
                await connection.commit()
                return True
            except BaseException:
                await connection.rollback()
                raise

    async def delete_memory(
        self,
        entry_id: str,
        scope: KnowledgeScope,
        *,
        effect_event_key: str | None = None,
    ) -> bool:
        async with self.database.operation():
            connection = self.database.connection
            await connection.execute("BEGIN IMMEDIATE")
            try:
                if effect_event_key is not None:
                    cursor = await connection.execute(
                        "SELECT 1 FROM command_effects WHERE event_key=?",
                        (effect_event_key,),
                    )
                    if await cursor.fetchone() is not None:
                        await connection.rollback()
                        return False
                await connection.execute(
                    "DELETE FROM memories_fts WHERE memory_id=? AND scope=?",
                    (entry_id, scope.key),
                )
                cursor = await connection.execute(
                    "DELETE FROM memories WHERE id=? AND scope=?", (entry_id, scope.key)
                )
                if effect_event_key is not None:
                    await connection.execute(
                        """
                        INSERT INTO command_effects(event_key, kind, created_at)
                        VALUES (?, 'memory_delete', ?)
                        """,
                        (effect_event_key, _now()),
                    )
                await connection.commit()
                return cursor.rowcount == 1
            except BaseException:
                await connection.rollback()
                raise

    async def rebuild_index(self) -> int:
        async with self.database.operation():
            connection = self.database.connection
            await connection.create_function("fts_document", 1, fts_document, deterministic=True)
            await connection.execute("BEGIN IMMEDIATE")
            try:
                await connection.execute("DELETE FROM chunks_fts")
                documents = await connection.execute(
                    """
                    INSERT INTO chunks_fts(chunk_id, scope, content)
                    SELECT CAST(id AS TEXT), scope, fts_document(content) FROM chunks
                    """
                )
                await connection.execute("DELETE FROM memories_fts")
                memories = await connection.execute(
                    """
                    INSERT INTO memories_fts(memory_id, scope, content)
                    SELECT id, scope, fts_document(content) FROM memories
                    """
                )
                await connection.commit()
                return documents.rowcount + memories.rowcount
            except BaseException:
                await connection.rollback()
                raise

    async def ingest(
        self,
        source: Path,
        scope: KnowledgeScope,
        *,
        prune: bool = False,
        dry_run: bool = True,
    ) -> IngestionReport:
        source = await asyncio.to_thread(lambda: source.expanduser().resolve())
        files = await asyncio.to_thread(self._discover, source)
        source_is_directory = await asyncio.to_thread(source.is_dir)
        if prune and not source_is_directory:
            raise ValueError("prune requires a directory source")
        parsed: list[ParsedDocument] = []
        skipped = 0
        root = source if source_is_directory else source.parent
        discovered_paths = {path.relative_to(root).as_posix() for path in files}
        for path in files:
            try:
                parsed.append(await asyncio.to_thread(self._parse, path, root, scope))
            except (OSError, UnicodeDecodeError, ValueError):
                skipped += 1
        if prune and skipped:
            raise ValueError("prune is refused when any discovered document was skipped")

        changed: list[ParsedDocument] = []
        unchanged = 0
        async with self.database.operation():
            for document in parsed:
                cursor = await self.database.connection.execute(
                    "SELECT revision FROM documents WHERE id=? AND scope=?",
                    (document.document_id, scope.key),
                )
                row = await cursor.fetchone()
                if row and row["revision"] == document.revision:
                    unchanged += 1
                else:
                    changed.append(document)
        stale = await self._stale_documents(root.as_uri(), scope, discovered_paths) if prune else []
        if dry_run:
            return IngestionReport(len(files), len(changed), unchanged, skipped, len(stale))

        for document in changed:
            await self._index_document(document, scope)
        deleted = await self._delete_documents(stale) if prune else 0
        return IngestionReport(len(files), len(changed), unchanged, skipped, deleted)

    def _discover(self, source: Path) -> list[Path]:
        if not source.exists():
            raise FileNotFoundError(source)
        if source.is_file():
            return [source] if source.suffix.casefold() in _ALLOWED_SUFFIXES else []
        result: list[Path] = []
        for path in source.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            if path.suffix.casefold() in _ALLOWED_SUFFIXES:
                result.append(path)
        return sorted(result)

    def _parse(self, path: Path, root: Path, scope: KnowledgeScope) -> ParsedDocument:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError("document is not a regular file")
            limit = self.settings.rag.max_document_bytes
            if file_stat.st_size > limit:
                raise ValueError("document exceeds configured size")
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = -1
                raw = handle.read(limit + 1)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if len(raw) > self.settings.rag.max_document_bytes:
            raise ValueError("document exceeds configured size")
        text = raw.decode("utf-8")
        title = path.stem
        if path.suffix.casefold() in {".html", ".htm"}:
            html_title, text = html_to_text(text)
            title = html_title or title
        relative = path.relative_to(root).as_posix()
        source_root = root.as_uri()
        content_hash = hashlib.sha256(raw).hexdigest()
        identity = f"{scope.key}\x1f{source_root}\x1f{relative}".encode()
        document_id = hashlib.blake2b(identity, digest_size=16).hexdigest()
        revision_source = (
            f"{content_hash}\x1f{self.settings.rag.chunk_chars}"
            f"\x1f{self.settings.rag.chunk_overlap_chars}"
            f"\x1f{INGESTION_PIPELINE_VERSION}\x1f{LEXICAL_PIPELINE_VERSION}"
        )
        revision = hashlib.sha256(revision_source.encode()).hexdigest()
        chunks = chunk_text(
            text,
            max_chars=self.settings.rag.chunk_chars,
            overlap_chars=self.settings.rag.chunk_overlap_chars,
            max_chunks=self.settings.rag.max_chunks_per_document,
        )
        if not chunks:
            raise ValueError("document contains no indexable text")
        return ParsedDocument(
            path=path,
            source_root=source_root,
            relative_path=relative,
            document_id=document_id,
            title=title,
            content=text,
            content_hash=content_hash,
            revision=revision,
            chunks=chunks,
        )

    async def _index_document(self, document: ParsedDocument, scope: KnowledgeScope) -> None:
        async with self.database.operation():
            connection = self.database.connection
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute(
                    "SELECT id FROM chunks WHERE document_id=?", (document.document_id,)
                )
                old_ids = [str(row[0]) for row in await cursor.fetchall()]
                for old_id in old_ids:
                    await connection.execute("DELETE FROM chunks_fts WHERE chunk_id=?", (old_id,))
                await connection.execute(
                    "DELETE FROM chunks WHERE document_id=?", (document.document_id,)
                )
                now = _now()
                await connection.execute(
                    """
                    INSERT INTO documents(
                        id, scope, source_root, source_uri, source_path, title,
                        content_hash, revision, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, ?)
                    ON CONFLICT(id) DO UPDATE SET source_root=excluded.source_root,
                        source_uri=excluded.source_uri, source_path=excluded.source_path,
                        title=excluded.title, content_hash=excluded.content_hash,
                        revision=excluded.revision, updated_at=excluded.updated_at
                    """,
                    (
                        document.document_id,
                        scope.key,
                        document.source_root,
                        document.path.as_uri(),
                        document.relative_path,
                        document.title,
                        document.content_hash,
                        document.revision,
                        now,
                        now,
                    ),
                )
                for chunk in document.chunks:
                    cursor = await connection.execute(
                        """
                        INSERT INTO chunks(
                            document_id, scope, ordinal, heading, content, content_hash,
                            metadata_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, '{}', ?)
                        """,
                        (
                            document.document_id,
                            scope.key,
                            chunk.ordinal,
                            chunk.heading,
                            chunk.content,
                            chunk.content_hash,
                            now,
                        ),
                    )
                    chunk_id = str(cursor.lastrowid)
                    await connection.execute(
                        "INSERT INTO chunks_fts(chunk_id, scope, content) VALUES (?, ?, ?)",
                        (chunk_id, scope.key, fts_document(chunk.content)),
                    )
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise

    async def _stale_documents(
        self, source_root: str, scope: KnowledgeScope, current_paths: set[str]
    ) -> list[str]:
        async with self.database.operation():
            cursor = await self.database.connection.execute(
                "SELECT id, source_path FROM documents WHERE scope=? AND source_root=?",
                (scope.key, source_root),
            )
            return [
                str(row["id"])
                for row in await cursor.fetchall()
                if row["source_path"] not in current_paths
            ]

    async def _delete_documents(self, document_ids: list[str]) -> int:
        if not document_ids:
            return 0
        async with self.database.operation():
            connection = self.database.connection
            await connection.execute("BEGIN IMMEDIATE")
            try:
                for document_id in document_ids:
                    chunk_cursor = await connection.execute(
                        "SELECT id FROM chunks WHERE document_id=?", (document_id,)
                    )
                    for chunk_row in await chunk_cursor.fetchall():
                        chunk_id = str(chunk_row["id"])
                        await connection.execute(
                            "DELETE FROM chunks_fts WHERE chunk_id=?", (chunk_id,)
                        )
                    await connection.execute("DELETE FROM documents WHERE id=?", (document_id,))
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return len(document_ids)

    async def stats(self) -> dict[str, Any]:
        return await self.database.database_stats()
