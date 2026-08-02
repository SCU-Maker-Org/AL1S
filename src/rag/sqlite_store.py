from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .documents import Chunk, Document
from .ingestion import UpsertResult


class SQLiteRagStore:
    """Map normalized RAG documents onto the application's SQLite schema."""

    def __init__(
        self,
        database_service: Any,
        *,
        collection: str = "technical_docs",
        default_trust_level: int = 50,
    ) -> None:
        normalized_collection = collection.strip()
        if not normalized_collection:
            raise ValueError("collection must not be empty")
        if not 0 <= default_trust_level <= 100:
            raise ValueError("default_trust_level must be between 0 and 100")
        self.database_service = database_service
        self.collection = normalized_collection
        self.default_trust_level = default_trust_level
        self._ingestion_source_root = ""
        self._ingestion_namespace = ""
        self._seen_document_keys: set[str] = set()

    def begin_ingestion(self, source_root: str, namespace: str) -> None:
        self._ingestion_source_root = source_root
        self._ingestion_namespace = namespace
        self._seen_document_keys = set()

    def finish_ingestion(self, *, prune: bool) -> dict[str, int]:
        result = {"documents": 0, "chunks": 0}
        if prune and self._ingestion_source_root:
            result = self.database_service.delete_rag_documents_not_seen(
                collection=self.collection,
                knowledge_namespace=self._ingestion_namespace,
                source_root=self._ingestion_source_root,
                seen_document_keys=sorted(self._seen_document_keys),
            )
        self._ingestion_source_root = ""
        self._ingestion_namespace = ""
        self._seen_document_keys = set()
        return result

    def upsert_document(
        self, document: Document, chunks: Sequence[Chunk]
    ) -> UpsertResult:
        previous = self._previous_chunks(document)
        document_record = self._document_record(document)
        chunk_records = [self._chunk_record(chunk) for chunk in chunks]
        result = self.database_service.upsert_rag_document(
            document_record, chunk_records
        )
        self._seen_document_keys.add(document_record["document_key"])

        if not result.get("changed", False):
            return UpsertResult.unchanged(len(chunks))
        if previous is None:
            return UpsertResult.inserted(len(chunks))

        current = {chunk.ordinal: chunk.content_sha256 for chunk in chunks}
        previous_ordinals = previous.keys()
        current_ordinals = current.keys()
        shared = previous_ordinals & current_ordinals
        inserted = len(current_ordinals - previous_ordinals)
        deleted = len(previous_ordinals - current_ordinals)
        unchanged = sum(previous[index] == current[index] for index in shared)
        updated = len(shared) - unchanged
        return UpsertResult(
            "updated",
            inserted_chunks=inserted,
            updated_chunks=updated,
            unchanged_chunks=unchanged,
            deleted_chunks=deleted,
        )

    def _previous_chunks(self, document: Document) -> dict[int, str] | None:
        with self.database_service.get_connection() as connection:
            row = connection.execute(
                """SELECT id FROM rag_documents
                   WHERE collection = ? AND knowledge_namespace = ? AND document_key = ?""",
                (
                    self.collection,
                    document.namespace,
                    _document_key(document, self._ingestion_source_root),
                ),
            ).fetchone()
            if row is None:
                return None
            rows = connection.execute(
                """SELECT chunk_index, content_hash FROM rag_chunks
                   WHERE document_id = ?""",
                (int(row["id"]),),
            ).fetchall()
        return {int(item["chunk_index"]): str(item["content_hash"]) for item in rows}

    def _document_record(self, document: Document) -> dict[str, Any]:
        metadata = dict(document.metadata)
        domains = list(document.domains)
        metadata.update(
            {
                "rag_document_id": document.document_id,
                "source_path": document.source_path,
                "source_sha256": document.source_sha256,
                "content_sha256": document.content_sha256,
                "revision_sha256": document.revision_sha256,
                "byte_size": document.byte_size,
                "format": document.format,
                "domains": domains,
            }
        )
        return {
            "collection": self.collection,
            "knowledge_namespace": document.namespace,
            "document_key": _document_key(document, self._ingestion_source_root),
            "source_root": self._ingestion_source_root,
            "source_uri": _source_uri(document, self._ingestion_source_root),
            "title": _metadata_text(
                document.metadata, "title", fallback=Path(document.source_path).stem
            ),
            "domain": domains[0] if domains else "",
            "subdomain": _metadata_text(document.metadata, "subdomain"),
            "product": _metadata_text(document.metadata, "product"),
            "version": _metadata_text(document.metadata, "version"),
            "language": _metadata_text(document.metadata, "language"),
            "license": _metadata_text(document.metadata, "license"),
            "trust_level": _trust_level(
                document.metadata.get("trust_level"), self.default_trust_level
            ),
            "content_hash": document.revision_sha256,
            "metadata": metadata,
        }

    @staticmethod
    def _chunk_record(chunk: Chunk) -> dict[str, Any]:
        metadata = {
            "rag_chunk_id": chunk.chunk_id,
            "source_path": chunk.source_path,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "heading_path": list(chunk.heading_path),
            "domains": list(chunk.domains),
        }
        return {
            "chunk_index": chunk.ordinal,
            "heading_path": " > ".join(chunk.heading_path),
            "content": chunk.content,
            "content_hash": chunk.content_sha256,
            "token_count": _estimated_token_count(chunk.content),
            "metadata": metadata,
        }


def _metadata_text(metadata: Mapping[str, Any], key: str, *, fallback: str = "") -> str:
    value = metadata.get(key, fallback)
    if value is None:
        return fallback
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    normalized = str(value).strip()
    return normalized or fallback


def _source_uri(document: Document, source_root: str = "") -> str:
    explicit = _metadata_text(
        document.metadata,
        "source_uri",
        fallback=_metadata_text(document.metadata, "source_url"),
    )
    if explicit:
        return explicit
    if source_root:
        return (Path(source_root) / document.source_path).resolve().as_uri()
    return document.source_path


def _document_key(document: Document, source_root: str = "") -> str:
    """Prefer a publisher-controlled ID; local documents fall back to their RAG ID."""
    for key in ("source_id", "document_id"):
        value = document.metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    if source_root:
        digest = hashlib.sha256(
            f"{source_root}\0{document.document_id}".encode("utf-8")
        ).hexdigest()
        return f"local:{digest}"
    return document.document_id


def _trust_level(value: Any, fallback: int) -> int:
    named_levels = {
        "official": 100,
        "verified": 90,
        "community": 60,
        "untrusted": 20,
    }
    if isinstance(value, str) and value.strip().casefold() in named_levels:
        return named_levels[value.strip().casefold()]
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return min(max(parsed, 0), 100)


def _estimated_token_count(content: str) -> int:
    if not content:
        return 0
    return max(1, math.ceil(len(content.encode("utf-8")) / 4))
