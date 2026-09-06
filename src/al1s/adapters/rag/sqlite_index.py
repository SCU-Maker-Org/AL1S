"""Scoped FTS5 plus sqlite-vec retrieval with reciprocal rank fusion."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlite_vec

from al1s.adapters.storage import Database
from al1s.config import AppSettings
from al1s.domain import KnowledgeScope
from al1s.domain.search_text import fts_document, search_terms
from al1s.domain.versions import INGESTION_PIPELINE_VERSION, LEXICAL_PIPELINE_VERSION
from al1s.ports.model import ModelClient
from al1s.ports.vector import KnowledgeHit

from .chunking import TextChunk, chunk_text
from .html import html_to_text

_ALLOWED_SUFFIXES = {".txt", ".md", ".markdown", ".html", ".htm"}
logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _rrf(rank: int, constant: int = 60) -> float:
    return 1.0 / (constant + rank)


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
    def __init__(self, database: Database, model: ModelClient, settings: AppSettings):
        self.database = database
        self.model = model
        self.settings = settings

    async def search(
        self,
        query: str,
        scope: KnowledgeScope,
        *,
        limit: int,
        include_memories: bool = True,
    ) -> tuple[KnowledgeHit, ...]:
        if not query.strip() or limit <= 0:
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
                await self._sparse_search(query, allowed_scope, include_memories=include_memories)
            )
        dense = []
        if await self.database.verify_vector_state():
            try:
                dense = await self._dense_search(query, scopes, include_memories=include_memories)
            except Exception:
                logger.exception("dense retrieval failed; using sparse results")
        if dense:
            rankings.append(dense)
        scores: dict[tuple[str, str], float] = {}
        records: dict[tuple[str, str], KnowledgeHit] = {}
        for ranking in rankings:
            for rank, hit in enumerate(ranking, start=1):
                key = (hit.source, hit.entry_id)
                records[key] = hit
                scores[key] = scores.get(key, 0.0) + _rrf(rank)
        ordered = sorted(scores, key=scores.__getitem__, reverse=True)
        result: list[KnowledgeHit] = []
        memory_count = 0
        for key in ordered:
            record = records[key]
            if record.source == "memory":
                if memory_count >= self.settings.memory.prompt_results:
                    continue
                memory_count += 1
            result.append(
                KnowledgeHit(
                    entry_id=record.entry_id,
                    content=record.content,
                    score=scores[key],
                    source=record.source,
                    metadata=record.metadata,
                )
            )
            if len(result) >= limit:
                break
        return tuple(result)

    async def _sparse_search(
        self,
        query: str,
        scope: KnowledgeScope,
        *,
        include_memories: bool,
    ) -> tuple[list[KnowledgeHit], ...]:
        expression = _fts_query(query)
        if not expression:
            return ()
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
            if include_memories and self.settings.memory.prompt_results > 0:
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
        return tuple(
            sorted(hits, key=lambda hit: hit.score, reverse=True)
            for hits in (document_hits, memory_hits)
            if hits
        )

    async def _dense_search(
        self,
        query: str,
        scopes: tuple[KnowledgeScope, ...],
        *,
        include_memories: bool,
    ) -> list[KnowledgeHit]:
        vectors = await self.model.embed((query,))
        if not vectors:
            return []
        packed = sqlite_vec.serialize_float32(list(vectors[0]))
        hits: list[KnowledgeHit] = []
        async with self.database.operation():
            generation = await self.database.active_vector_generation()
            for scope in scopes:
                kinds = ("chunks", "memories") if include_memories else ("chunks",)
                for kind in kinds:
                    table = f"{kind}_vec_v{generation}"
                    cursor = await self.database.connection.execute(
                        f"""SELECT entity_id, distance FROM {table}
                        WHERE embedding MATCH ? AND k=? AND scope=?
                        ORDER BY distance""",  # noqa: S608
                        (packed, self.settings.rag.dense_candidates, scope.key),
                    )
                    rows = await cursor.fetchall()
                    hits.extend(await self._hydrate_dense(kind, rows, scope))
        ordered = sorted(hits, key=lambda hit: hit.score, reverse=True)
        memory_limit = self.settings.memory.prompt_results
        memory_count = 0
        limited: list[KnowledgeHit] = []
        for hit in ordered:
            if hit.source == "memory":
                if memory_count >= memory_limit:
                    continue
                memory_count += 1
            limited.append(hit)
        return limited

    async def _hydrate_dense(
        self, kind: str, rows: list[Any], scope: KnowledgeScope
    ) -> list[KnowledgeHit]:
        hits: list[KnowledgeHit] = []
        for row in rows:
            entity_id = str(row["entity_id"])
            if kind == "chunks":
                cursor = await self.database.connection.execute(
                    """
                    SELECT c.content, c.heading, d.source_uri
                    FROM chunks c JOIN documents d
                      ON d.id=c.document_id AND d.scope=c.scope
                    WHERE c.id=? AND c.scope=?
                    """,
                    (int(entity_id), scope.key),
                )
                record = await cursor.fetchone()
                if record:
                    hits.append(
                        KnowledgeHit(
                            entry_id=entity_id,
                            content=record["content"],
                            score=1.0 / (1.0 + float(row["distance"])),
                            source="document",
                            metadata={
                                "heading": record["heading"],
                                "source_uri": record["source_uri"],
                            },
                        )
                    )
            else:
                cursor = await self.database.connection.execute(
                    "SELECT content, category FROM memories WHERE id=? AND scope=?",
                    (entity_id, scope.key),
                )
                record = await cursor.fetchone()
                if record:
                    hits.append(
                        KnowledgeHit(
                            entry_id=entity_id,
                            content=record["content"],
                            score=1.0 / (1.0 + float(row["distance"])),
                            source="memory",
                            metadata={"category": record["category"]},
                        )
                    )
        return hits

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
        vector = None
        if await self.database.verify_vector_state():
            try:
                vectors = await self.model.embed((content,))
                if len(vectors) != 1:
                    raise RuntimeError("embedding provider returned an unexpected vector count")
                vector = sqlite_vec.serialize_float32(list(vectors[0]))
            except Exception:
                logger.exception("memory embedding failed; storing sparse memory")
        async with self.database.operation():
            connection = self.database.connection
            generation = await self.database.active_vector_generation()
            table = f"memories_vec_v{generation}"
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
                if vector is not None:
                    await connection.execute(
                        f"DELETE FROM {table} WHERE entity_id=? AND scope=?",  # noqa: S608
                        (entry_id, scope.key),
                    )
                    await connection.execute(
                        f"INSERT INTO {table}(entity_id, scope, embedding) VALUES (?, ?, ?)",  # noqa: S608
                        (entry_id, scope.key, vector),
                    )
                else:
                    await self._mark_vector_dirty(connection)
                await self._bump_knowledge_revision(connection)
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
        vector_ready = await self.database.verify_vector_state()
        async with self.database.operation():
            connection = self.database.connection
            generation = await self.database.active_vector_generation()
            table = f"memories_vec_v{generation}"
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
                if vector_ready:
                    await connection.execute(
                        f"DELETE FROM {table} WHERE entity_id=? AND scope=?",  # noqa: S608
                        (entry_id, scope.key),
                    )
                else:
                    await self._mark_vector_dirty(connection)
                await connection.execute(
                    "DELETE FROM memories_fts WHERE memory_id=? AND scope=?",
                    (entry_id, scope.key),
                )
                cursor = await connection.execute(
                    "DELETE FROM memories WHERE id=? AND scope=?", (entry_id, scope.key)
                )
                if cursor.rowcount == 1:
                    await self._bump_knowledge_revision(connection)
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

    async def rebuild_vectors(self) -> int:
        if not self.database.vector_extension_loaded:
            raise RuntimeError("sqlite-vec extension is unavailable")
        async with self.database.vector_rebuild_operation():
            source_revision = await self._knowledge_revision()
            async with self.database.operation():
                connection = self.database.connection
                old_generation = await self.database.active_vector_generation()
                generation = old_generation + 1
                dimensions = self.settings.models.embedding.dimensions
                await connection.execute("BEGIN IMMEDIATE")
                try:
                    await self._drop_vector_generation(connection, generation)
                    for kind in ("chunks", "memories"):
                        await connection.execute(
                            f"CREATE VIRTUAL TABLE {kind}_vec_v{generation} "
                            "USING vec0(entity_id text primary key, scope text, "
                            f"embedding float[{dimensions}])"
                        )
                    await connection.commit()
                except BaseException:
                    await connection.rollback()
                    raise
            try:
                total = 0
                batch_size = self.settings.models.embedding.batch_size
                for kind in ("chunks", "memories"):
                    offset = 0
                    while rows := await self._vector_source_batch(kind, offset, batch_size):
                        embeddings = await self.model.embed(tuple(row["content"] for row in rows))
                        if len(embeddings) != len(rows):
                            raise RuntimeError(
                                "embedding provider returned an unexpected vector count"
                            )
                        packed = [
                            (
                                str(row["id"]),
                                row["scope"],
                                sqlite_vec.serialize_float32(list(embedding)),
                            )
                            for row, embedding in zip(rows, embeddings, strict=True)
                        ]
                        async with self.database.operation():
                            connection = self.database.connection
                            await connection.execute("BEGIN IMMEDIATE")
                            try:
                                await connection.executemany(
                                    f"INSERT INTO {kind}_vec_v{generation}("  # noqa: S608
                                    "entity_id, scope, embedding) VALUES (?, ?, ?)",
                                    packed,
                                )
                                await connection.commit()
                            except BaseException:
                                await connection.rollback()
                                raise
                        total += len(rows)
                        offset += len(rows)
                async with self.database.operation():
                    connection = self.database.connection
                    await connection.execute("BEGIN IMMEDIATE")
                    try:
                        if await self._knowledge_revision_locked() != source_revision:
                            raise RuntimeError(
                                "knowledge changed during vector rebuild; retry the rebuild"
                            )
                        now = _now()
                        for key, value in (
                            ("vector_generation", str(generation)),
                            ("vector_dimensions", str(dimensions)),
                            ("vector_fingerprint", self.database.vector_fingerprint),
                            ("vector_dirty", "0"),
                        ):
                            await connection.execute(
                                """
                                INSERT INTO app_metadata(key, value, updated_at) VALUES (?, ?, ?)
                                ON CONFLICT(key) DO UPDATE SET
                                    value=excluded.value, updated_at=excluded.updated_at
                                """,
                                (key, value, now),
                            )
                        await connection.execute(
                            "UPDATE chunks SET vector_generation=?", (generation,)
                        )
                        await self._drop_inactive_vector_generations(connection, generation)
                        await connection.commit()
                    except BaseException:
                        await connection.rollback()
                        raise
                self.database.mark_vector_ready()
                return total
            except BaseException:
                async with self.database.operation():
                    connection = self.database.connection
                    await connection.execute("BEGIN IMMEDIATE")
                    try:
                        active_generation = await self.database.active_vector_generation()
                        if active_generation != generation:
                            await self._drop_vector_generation(connection, generation)
                        await connection.commit()
                    except BaseException:
                        await connection.rollback()
                        logger.exception("failed to remove incomplete vector generation")
                raise

    async def _vector_source_batch(self, kind: str, offset: int, limit: int) -> list[Any]:
        if kind not in {"chunks", "memories"}:
            raise ValueError("invalid vector source")
        async with self.database.operation():
            cursor = await self.database.connection.execute(
                f"SELECT id, scope, content FROM {kind} ORDER BY id LIMIT ? OFFSET ?",  # noqa: S608
                (limit, offset),
            )
            return list(await cursor.fetchall())

    async def _knowledge_revision(self) -> int:
        async with self.database.operation():
            return await self._knowledge_revision_locked()

    async def _knowledge_revision_locked(self) -> int:
        cursor = await self.database.connection.execute(
            "SELECT value FROM app_metadata WHERE key='knowledge_revision'"
        )
        row = await cursor.fetchone()
        return int(row["value"]) if row else 0

    @staticmethod
    async def _drop_vector_generation(connection: Any, generation: int) -> None:
        for kind in ("chunks", "memories"):
            await connection.execute(f"DROP TABLE IF EXISTS {kind}_vec_v{generation}")

    @staticmethod
    async def _drop_inactive_vector_generations(connection: Any, active_generation: int) -> None:
        cursor = await connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table'
              AND (name LIKE 'chunks_vec_v%' OR name LIKE 'memories_vec_v%')
            """
        )
        active = {
            f"chunks_vec_v{active_generation}",
            f"memories_vec_v{active_generation}",
        }
        for row in await cursor.fetchall():
            name = str(row["name"])
            if name in active or not re.fullmatch(
                r"(?:chunks|memories)_vec_v[1-9][0-9]{0,8}", name
            ):
                continue
            await connection.execute(f"DROP TABLE {name}")

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
        vector_ready = await self.database.verify_vector_state()
        packed_embeddings: list[bytes | None] = []
        if vector_ready:
            try:
                batch_size = self.settings.models.embedding.batch_size
                for start in range(0, len(document.chunks), batch_size):
                    batch = document.chunks[start : start + batch_size]
                    embeddings = await self.model.embed(tuple(chunk.content for chunk in batch))
                    if len(embeddings) != len(batch):
                        raise RuntimeError("embedding provider returned an unexpected vector count")
                    packed_embeddings.extend(
                        sqlite_vec.serialize_float32(list(embedding)) for embedding in embeddings
                    )
            except Exception:
                logger.exception("document embedding failed; storing sparse index")
                vector_ready = False
                packed_embeddings.clear()
        if not vector_ready:
            packed_embeddings.extend(None for _ in document.chunks)
        async with self.database.operation():
            generation = await self.database.active_vector_generation()
            vector_table = f"chunks_vec_v{generation}"
            connection = self.database.connection
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute(
                    "SELECT id FROM chunks WHERE document_id=?", (document.document_id,)
                )
                old_ids = [str(row[0]) for row in await cursor.fetchall()]
                for old_id in old_ids:
                    await connection.execute("DELETE FROM chunks_fts WHERE chunk_id=?", (old_id,))
                    if vector_ready:
                        await connection.execute(
                            f"DELETE FROM {vector_table} WHERE entity_id=?",  # noqa: S608
                            (old_id,),
                        )
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
                for chunk, embedding in zip(document.chunks, packed_embeddings, strict=True):
                    cursor = await connection.execute(
                        """
                        INSERT INTO chunks(
                            document_id, scope, ordinal, heading, content, content_hash,
                            metadata_json, vector_generation, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, '{}', ?, ?)
                        """,
                        (
                            document.document_id,
                            scope.key,
                            chunk.ordinal,
                            chunk.heading,
                            chunk.content,
                            chunk.content_hash,
                            generation if vector_ready else None,
                            now,
                        ),
                    )
                    chunk_id = str(cursor.lastrowid)
                    await connection.execute(
                        "INSERT INTO chunks_fts(chunk_id, scope, content) VALUES (?, ?, ?)",
                        (chunk_id, scope.key, fts_document(chunk.content)),
                    )
                    if vector_ready:
                        assert embedding is not None
                        await connection.execute(
                            f"INSERT INTO {vector_table}(entity_id, scope, embedding) "  # noqa: S608
                            "VALUES (?, ?, ?)",
                            (
                                chunk_id,
                                scope.key,
                                embedding,
                            ),
                        )
                if not vector_ready:
                    await self._mark_vector_dirty(connection)
                await self._bump_knowledge_revision(connection)
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
        vector_ready = await self.database.verify_vector_state()
        async with self.database.operation():
            connection = self.database.connection
            generation = await self.database.active_vector_generation()
            vector_table = f"chunks_vec_v{generation}"
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
                        if vector_ready:
                            await connection.execute(
                                f"DELETE FROM {vector_table} WHERE entity_id=?",  # noqa: S608
                                (chunk_id,),
                            )
                    await connection.execute("DELETE FROM documents WHERE id=?", (document_id,))
                if not vector_ready:
                    await self._mark_vector_dirty(connection)
                if document_ids:
                    await self._bump_knowledge_revision(connection)
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise
        return len(document_ids)

    @staticmethod
    async def _mark_vector_dirty(connection) -> None:
        await connection.execute(
            """
            INSERT INTO app_metadata(key, value, updated_at) VALUES ('vector_dirty', '1', ?)
            ON CONFLICT(key) DO UPDATE SET value='1', updated_at=excluded.updated_at
            """,
            (_now(),),
        )

    @staticmethod
    async def _bump_knowledge_revision(connection) -> None:
        await connection.execute(
            """
            INSERT INTO app_metadata(key, value, updated_at)
            VALUES ('knowledge_revision', '1', ?)
            ON CONFLICT(key) DO UPDATE SET
                value=CAST(app_metadata.value AS INTEGER) + 1,
                updated_at=excluded.updated_at
            """,
            (_now(),),
        )

    async def stats(self) -> dict[str, Any]:
        stats = await self.database.database_stats()
        stats["vector_enabled"] = self.database.vector_enabled
        stats["vector_rebuild_required"] = self.database.vector_rebuild_required
        stats["embedding_dimensions"] = self.settings.models.embedding.dimensions
        stats["vector_generation"] = await self.database.active_vector_generation()
        return stats
