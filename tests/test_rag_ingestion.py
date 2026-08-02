from __future__ import annotations

import pytest

from src.rag import (
    ChunkingConfig,
    DocumentDecodeError,
    IngestionConfig,
    InvalidDomainError,
    RagIngestor,
    UpsertResult,
    chunk_document,
    discover_files,
    is_supported_document,
    load_document,
)


class MemoryUpsert:
    def __init__(self):
        self.documents: dict[str, tuple[str, dict[str, str]]] = {}

    def upsert_document(self, document, chunks):
        current = {chunk.chunk_id: chunk.content_sha256 for chunk in chunks}
        previous = self.documents.get(document.document_id)
        self.documents[document.document_id] = (document.revision_sha256, current)
        if previous is None:
            return UpsertResult.inserted(len(chunks))
        previous_revision, previous_chunks = previous
        if previous_revision == document.revision_sha256 and previous_chunks == current:
            return UpsertResult.unchanged(len(chunks))
        inserted = len(current.keys() - previous_chunks.keys())
        deleted = len(previous_chunks.keys() - current.keys())
        shared = current.keys() & previous_chunks.keys()
        updated = sum(current[key] != previous_chunks[key] for key in shared)
        unchanged = len(shared) - updated
        return UpsertResult(
            "updated",
            inserted_chunks=inserted,
            updated_chunks=updated,
            unchanged_chunks=unchanged,
            deleted_chunks=deleted,
        )


def test_loads_frontmatter_and_stable_document_identity(tmp_path):
    source = tmp_path / "postgres.md"
    source.write_text(
        """---
title: PostgreSQL internals
domains: [db, sys, storage]
owner: alice
---
# WAL

Write-ahead logging preserves durability.
""",
        encoding="utf-8",
    )

    first = load_document(source, root=tmp_path, namespace="computer-science")
    second = load_document(source, root=tmp_path, namespace="computer-science")

    assert first == second
    assert first.source_path == "postgres.md"
    assert first.domains == ("sys", "db", "storage")
    assert first.metadata["title"] == "PostgreSQL internals"
    assert first.metadata["owner"] == "alice"
    assert first.content.startswith("# WAL")
    assert (
        len(first.document_id)
        == len(first.content_sha256)
        == len(first.source_sha256)
        == len(first.revision_sha256)
        == 64
    )

    previous_id = first.document_id
    previous_hash = first.content_sha256
    source.write_text(
        source.read_text() + "Checkpointing is related.\n", encoding="utf-8"
    )
    changed = load_document(source, root=tmp_path, namespace="computer-science")
    assert changed.document_id == previous_id
    assert changed.content_sha256 != previous_hash


def test_rejects_domain_outside_controlled_vocabulary(tmp_path):
    source = tmp_path / "bad.md"
    source.write_text("---\ndomains: [sys, crypto]\n---\n# Bad\n", encoding="utf-8")

    with pytest.raises(InvalidDomainError, match="crypto"):
        load_document(source, root=tmp_path)


def test_frontmatter_domains_override_cli_defaults(tmp_path):
    source = tmp_path / "planner.md"
    source.write_text(
        "---\ndomains: [db, compile]\n---\n# Planner\nQuery planning.",
        encoding="utf-8",
    )

    document = load_document(
        source,
        root=tmp_path,
        default_domains="sys,hpc,storage",
    )

    assert document.domains == ("compile", "db")


def test_rejects_binary_content_with_supported_extension(tmp_path):
    source = tmp_path / "binary.txt"
    source.write_bytes(b"prefix\x00suffix")

    with pytest.raises(DocumentDecodeError, match="Binary document"):
        load_document(source, root=tmp_path)


def test_heading_aware_chunks_have_overlap_and_stable_ids(tmp_path):
    source = tmp_path / "systems.md"
    source.write_text(
        "# Memory\n\n"
        + ("virtual memory page table translation. " * 16)
        + "\n\n## UFFD\n\n"
        + ("userfaultfd missing-page handling. " * 14),
        encoding="utf-8",
    )
    document = load_document(source, root=tmp_path, default_domains=["sys"])
    config = ChunkingConfig(max_chars=180, overlap_chars=30)

    first = chunk_document(document, config)
    second = chunk_document(document, config)

    assert first == second
    assert len(first) > 3
    assert all(len(chunk.content) <= 180 for chunk in first)
    assert [chunk.ordinal for chunk in first] == list(range(len(first)))
    assert first[0].heading_path == ("Memory",)
    assert any(chunk.heading_path == ("Memory", "UFFD") for chunk in first)
    assert all(chunk.domains == ("sys",) for chunk in first)
    assert all(
        len(chunk.chunk_id) == len(chunk.content_sha256) == 64 for chunk in first
    )
    assert any(
        earlier.content[-20:] in later.content
        for earlier, later in zip(first, first[1:])
        if earlier.heading_path == later.heading_path
    )


def test_markdown_code_fence_is_not_treated_as_a_heading(tmp_path):
    source = tmp_path / "commands.md"
    source.write_text(
        """# Real heading

```bash
# This is a shell comment, not a heading
echo ok
```
""",
        encoding="utf-8",
    )

    chunks = chunk_document(load_document(source, root=tmp_path))

    assert {chunk.heading_path for chunk in chunks} == {("Real heading",)}


def test_common_documents_and_code_are_supported(tmp_path):
    for name in (
        "notes.txt",
        "manual.rst",
        "page.html",
        "data.json",
        "config.yaml",
        "settings.toml",
        "kernel.cu",
        "server.py",
        "query.sql",
        "Dockerfile",
    ):
        path = tmp_path / name
        path.write_text("plain technical content", encoding="utf-8")

    assert all(
        is_supported_document(tmp_path / name)
        for name in (
            "notes.txt",
            "manual.rst",
            "page.html",
            "data.json",
            "config.yaml",
            "settings.toml",
            "kernel.cu",
            "server.py",
            "query.sql",
            "Dockerfile",
        )
    )


def test_html_title_and_headings_are_preserved(tmp_path):
    source = tmp_path / "virtio.html"
    source.write_text(
        """<html><head><title>Virtio</title><script>ignore()</script></head>
        <body><h1>Queues</h1><p>Descriptor rings.</p></body></html>""",
        encoding="utf-8",
    )

    document = load_document(source, root=tmp_path, default_domains="sys")
    chunks = chunk_document(document)

    assert document.metadata["title"] == "Virtio"
    assert "ignore()" not in document.content
    assert "# Queues" in document.content
    assert chunks[0].heading_path == ("Queues",)


def test_discovery_enforces_size_and_symlink_boundary(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "ok.md").write_text("# Safe\ncontent", encoding="utf-8")
    (corpus / "large.txt").write_text("x" * 100, encoding="utf-8")
    (corpus / "image.bin").write_bytes(b"not a document")
    outside = tmp_path / "secret.md"
    outside.write_text("secret", encoding="utf-8")
    (corpus / "escape.md").symlink_to(outside)
    (corpus / "alias.md").symlink_to(corpus / "ok.md")

    result = discover_files(corpus, max_file_bytes=64)

    assert [item.source_path for item in result.files] == ["ok.md"]
    assert result.discovered_files == 5
    assert result.oversized_files == 1
    assert result.unsupported_files == 1
    assert result.unsafe_symlinks == 1
    assert result.duplicate_files == 1
    assert any(issue.reason == "external_symlink" for issue in result.issues)


@pytest.mark.asyncio
async def test_ingestion_reports_idempotent_upserts(tmp_path):
    source = tmp_path / "training.md"
    source.write_text(
        "# Collectives\n\n" + "All-reduce coordinates GPU workers. " * 24,
        encoding="utf-8",
    )
    store = MemoryUpsert()
    ingestor = RagIngestor(
        IngestionConfig(chunking=ChunkingConfig(max_chars=120, overlap_chars=20))
    )

    first = await ingestor.ingest(
        tmp_path,
        store,
        namespace="ai-systems",
        default_domains=["hpc", "distributed", "ai_workload"],
    )
    second = await ingestor.ingest(
        tmp_path,
        store,
        namespace="ai-systems",
        default_domains=["hpc", "distributed", "ai_workload"],
    )
    source.write_text(
        source.read_text(encoding="utf-8") + "\nNCCL uses topology-aware rings.\n",
        encoding="utf-8",
    )
    third = await ingestor.ingest(
        tmp_path,
        store,
        namespace="ai-systems",
        default_domains=["hpc", "distributed", "ai_workload"],
    )

    assert first.inserted_documents == 1
    assert first.inserted_chunks == first.chunks_seen > 1
    assert second.unchanged_documents == 1
    assert second.unchanged_chunks == second.chunks_seen == first.chunks_seen
    assert second.inserted_chunks == second.updated_chunks == 0
    assert second.processed_documents == 1
    assert third.updated_documents == 1
    assert third.updated_chunks + third.inserted_chunks > 0


@pytest.mark.asyncio
async def test_ingestion_accepts_async_callback(tmp_path):
    (tmp_path / "network.txt").write_text("TCP congestion control", encoding="utf-8")
    seen: list[str] = []

    async def upsert(document, chunks):
        seen.append(document.source_path)
        return UpsertResult.inserted(len(chunks))

    stats = await RagIngestor().ingest(tmp_path, upsert, default_domains="sys")

    assert seen == ["network.txt"]
    assert stats.inserted_documents == 1
    assert stats.failed_files == 0


@pytest.mark.asyncio
async def test_ingestion_skips_documents_marked_not_for_indexing(tmp_path):
    (tmp_path / "policy.md").write_text(
        "---\nindex: false\ndomains: [sys]\n---\n# Corpus policy\nNot facts.",
        encoding="utf-8",
    )
    seen = []

    def upsert(document, chunks):
        seen.append(document.source_path)
        return UpsertResult.inserted(len(chunks))

    stats = await RagIngestor().ingest(tmp_path, upsert)

    assert seen == []
    assert stats.excluded_documents == 1
    assert stats.processed_documents == 0


@pytest.mark.asyncio
async def test_ingestion_rejects_unclassified_documents(tmp_path):
    (tmp_path / "unknown.md").write_text("# Unknown\nFacts", encoding="utf-8")

    stats = await RagIngestor().ingest(
        tmp_path,
        lambda document, chunks: UpsertResult.inserted(len(chunks)),
    )

    assert stats.failed_files == 1
    assert stats.processed_documents == 0
    assert stats.issues[0].reason == "missing_domains"


@pytest.mark.asyncio
async def test_strict_ingestion_rejects_discovery_issues(tmp_path):
    (tmp_path / "valid.md").write_text("# Valid\nFacts", encoding="utf-8")
    (tmp_path / "oversized.md").write_text("x" * 128, encoding="utf-8")
    ingestor = RagIngestor(IngestionConfig(max_file_bytes=64, strict=True))

    with pytest.raises(ValueError, match="Strict discovery rejected.*file_too_large"):
        await ingestor.ingest(
            tmp_path,
            lambda document, chunks: UpsertResult.inserted(len(chunks)),
            default_domains="sys",
        )
