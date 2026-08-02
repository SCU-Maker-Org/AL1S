from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from src.infra.database import DatabaseService
from src.rag import ChunkingConfig, RagIngestor, chunk_document, load_document
from src.rag.sqlite_store import SQLiteRagStore


def _database(tmp_path: Path) -> DatabaseService:
    path = tmp_path / "rag.db"
    init_sql = (Path(__file__).parents[1] / "data" / "init_db.sql").read_text(
        encoding="utf-8"
    )
    with sqlite3.connect(path) as connection:
        connection.executescript(init_sql)
    return DatabaseService(str(path))


def _load(source: Path):
    document = load_document(
        source,
        root=source.parent,
        namespace="global:technical",
        default_domains="db,sys",
    )
    chunks = chunk_document(document, ChunkingConfig(max_chars=120, overlap_chars=20))
    return document, chunks


def test_sqlite_store_inserts_and_maps_document_metadata(tmp_path):
    source = tmp_path / "postgres.md"
    source.write_text(
        """---
title: PostgreSQL WAL
product: PostgreSQL
version: "18"
license: PostgreSQL
language: en
subdomain: wal
trust_level: 95
source_uri: https://www.postgresql.org/docs/18/wal-intro.html
source_id: postgresql_wal
---
# WAL

Write-ahead logging makes data changes durable before heap pages are flushed.
""",
        encoding="utf-8",
    )
    database = _database(tmp_path)
    store = SQLiteRagStore(database, collection="computer_systems")
    document, chunks = _load(source)

    result = store.upsert_document(document, chunks)

    assert result.document_status == "inserted"
    assert result.inserted_chunks == len(chunks)
    with database.get_connection() as connection:
        stored_document = connection.execute("SELECT * FROM rag_documents").fetchone()
        stored_chunks = connection.execute(
            "SELECT * FROM rag_chunks ORDER BY chunk_index"
        ).fetchall()
    assert stored_document["collection"] == "computer_systems"
    assert stored_document["knowledge_namespace"] == "global:technical"
    assert stored_document["document_key"] == "postgresql_wal"
    assert (
        stored_document["source_uri"]
        == "https://www.postgresql.org/docs/18/wal-intro.html"
    )
    assert stored_document["title"] == "PostgreSQL WAL"
    assert stored_document["domain"] == "sys"
    assert stored_document["subdomain"] == "wal"
    assert stored_document["product"] == "PostgreSQL"
    assert stored_document["version"] == "18"
    assert stored_document["license"] == "PostgreSQL"
    assert stored_document["language"] == "en"
    assert stored_document["trust_level"] == 95
    metadata = json.loads(stored_document["metadata_json"])
    assert metadata["domains"] == ["sys", "db"]
    assert metadata["rag_document_id"] == document.document_id
    assert metadata["source_sha256"] == document.source_sha256
    assert len(stored_chunks) == len(chunks)
    chunk_metadata = json.loads(stored_chunks[0]["metadata_json"])
    assert chunk_metadata["rag_chunk_id"] == chunks[0].chunk_id
    assert chunk_metadata["source_path"] == "postgres.md"
    assert stored_chunks[0]["heading_path"] == "WAL"
    assert stored_chunks[0]["token_count"] > 0


def test_sqlite_store_is_idempotent(tmp_path):
    source = tmp_path / "virtio.md"
    source.write_text(
        "# Virtio\n\nA virtqueue connects guest and host.\n", encoding="utf-8"
    )
    database = _database(tmp_path)
    store = SQLiteRagStore(database)
    document, chunks = _load(source)

    first = store.upsert_document(document, chunks)
    second = store.upsert_document(document, chunks)

    assert first.document_status == "inserted"
    assert second.document_status == "unchanged"
    assert second.unchanged_chunks == len(chunks)
    assert database.get_rag_document_stats() == {
        "documents": 1,
        "chunks": len(chunks),
    }


def test_official_trust_level_maps_to_highest_numeric_score(tmp_path):
    source = tmp_path / "official.md"
    source.write_text(
        "---\ndomains: [sys]\ntrust_level: official\n---\n# Kernel\nOfficial docs.",
        encoding="utf-8",
    )
    database = _database(tmp_path)
    store = SQLiteRagStore(database)
    document, chunks = _load(source)

    store.upsert_document(document, chunks)

    with database.get_connection() as connection:
        trust_level = connection.execute(
            "SELECT trust_level FROM rag_documents"
        ).fetchone()[0]
    assert trust_level == 100


def test_sqlite_store_reports_updates_and_replaces_fts_rows(tmp_path):
    source = tmp_path / "planner.md"
    source.write_text(
        "# Planner\n\n" + ("Sequential scan behavior. " * 12), encoding="utf-8"
    )
    database = _database(tmp_path)
    store = SQLiteRagStore(database)
    first_document, first_chunks = _load(source)
    store.upsert_document(first_document, first_chunks)

    source.write_text(
        "# Planner\n\n" + ("Bitmap index scan behavior. " * 5), encoding="utf-8"
    )
    second_document, second_chunks = _load(source)
    result = store.upsert_document(second_document, second_chunks)

    assert result.document_status == "updated"
    assert (
        result.updated_chunks + result.inserted_chunks + result.unchanged_chunks
        == len(second_chunks)
    )
    assert result.deleted_chunks == max(0, len(first_chunks) - len(second_chunks))
    assert database.get_rag_document_stats() == {
        "documents": 1,
        "chunks": len(second_chunks),
    }
    lexical = database.search_rag_chunks_lexical(
        "Bitmap index scan",
        collections=["technical_docs"],
        knowledge_namespaces=["global:technical"],
    )
    assert lexical
    assert "Bitmap" in lexical[0]["content"]
    assert not database.search_rag_chunks_lexical(
        "Sequential",
        collections=["technical_docs"],
        knowledge_namespaces=["global:technical"],
    )


def test_source_id_updates_a_document_when_its_official_url_changes(tmp_path):
    source = tmp_path / "postgres.md"
    source.write_text(
        """---
source_id: postgresql_mvcc
source_uri: https://www.postgresql.org/docs/17/mvcc.html
domains: [db, sys]
---
# MVCC

Version 17 visibility rules.
""",
        encoding="utf-8",
    )
    database = _database(tmp_path)
    store = SQLiteRagStore(database)
    first_document, first_chunks = _load(source)
    store.upsert_document(first_document, first_chunks)

    source.write_text(
        """---
source_id: postgresql_mvcc
source_uri: https://www.postgresql.org/docs/18/mvcc.html
domains: [db, sys]
---
# MVCC

Version 18 visibility rules and snapshots.
""",
        encoding="utf-8",
    )
    second_document, second_chunks = _load(source)
    result = store.upsert_document(second_document, second_chunks)

    assert result.document_status == "updated"
    with database.get_connection() as connection:
        rows = connection.execute(
            "SELECT document_key, source_uri FROM rag_documents"
        ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("postgresql_mvcc", "https://www.postgresql.org/docs/18/mvcc.html")
    ]


def test_chinese_terms_are_recalled_by_fts_bigrams(tmp_path):
    source = tmp_path / "transactions.md"
    source.write_text(
        "---\ndomains: [db]\n---\n# 事务\n事务提供原子性，索引支持快速查询。",
        encoding="utf-8",
    )
    database = _database(tmp_path)
    store = SQLiteRagStore(database)
    document, chunks = _load(source)
    store.upsert_document(document, chunks)

    transaction_results = database.search_rag_chunks_lexical("事务")
    combined_results = database.search_rag_chunks_lexical("事务 索引")

    assert transaction_results
    assert combined_results
    assert "事务提供原子性" in transaction_results[0]["content"]


@pytest.mark.asyncio
async def test_ingestion_prunes_deleted_and_excluded_files_within_only_its_root(
    tmp_path,
):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_document = first_root / "notes.md"
    second_document = second_root / "notes.md"
    first_document.write_text("# First\nKernel memory.", encoding="utf-8")
    second_document.write_text("# Second\nDatabase indexes.", encoding="utf-8")
    database = _database(tmp_path)
    store = SQLiteRagStore(database)
    ingestor = RagIngestor()

    await ingestor.ingest(first_root, store, default_domains="sys")
    await ingestor.ingest(second_root, store, default_domains="db")
    assert database.get_rag_document_stats()["documents"] == 2

    first_document.unlink()
    deleted = await ingestor.ingest(first_root, store, default_domains="sys")
    assert deleted.deleted_documents == 1
    assert database.get_rag_document_stats()["documents"] == 1

    second_document.write_text(
        "---\nindex: false\ndomains: [db]\n---\n# Excluded\nOld facts.",
        encoding="utf-8",
    )
    excluded = await ingestor.ingest(second_root, store, default_domains="db")
    assert excluded.excluded_documents == 1
    assert excluded.deleted_documents == 1
    assert database.get_rag_document_stats() == {"documents": 0, "chunks": 0}
