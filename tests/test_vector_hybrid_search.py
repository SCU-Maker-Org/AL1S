from __future__ import annotations

import pytest

from src.infra.vector import VectorService


class FakeEmbedding:
    async def encode_single(self, text, *, is_query=False):
        assert text == "WAL visibility"
        assert is_query is True
        return [1.0, 0.0]


class FakeStore:
    size = 4

    def __init__(self):
        self.search_sizes = []

    async def search(self, query_vector, top_k, threshold):
        self.search_sizes.append(top_k)
        return [
            (
                0,
                0.99,
                {
                    "id": 900,
                    "title": "other user",
                    "content": "must stay isolated",
                    "knowledge_namespace": "private:2",
                    "record_type": "conversation_memory",
                },
            ),
            (
                1,
                0.80,
                {
                    "id": "rag:10",
                    "chunk_id": 10,
                    "title": "PostgreSQL WAL",
                    "content": "dense document",
                    "knowledge_namespace": "global:technical",
                    "collection": "technical_docs",
                    "record_type": "document_chunk",
                },
            ),
            (
                2,
                0.70,
                {
                    "id": 101,
                    "title": "owner memory",
                    "content": "private note",
                    "knowledge_namespace": "private:1",
                    "record_type": "conversation_memory",
                },
            ),
        ]


class FakeDatabase:
    def __init__(self):
        self.calls = []

    def search_rag_chunks_lexical(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [
            {
                "chunk_id": 11,
                "title": "PostgreSQL MVCC",
                "content": "lexical document",
                "knowledge_namespace": "global:technical",
                "collection": "technical_docs",
                "domain": "db",
                "trust_level": 95,
            }
        ]


class LargeSparseStore:
    size = 1_000_000

    def __init__(self):
        self.search_sizes = []

    async def search(self, query_vector, top_k, threshold):
        self.search_sizes.append(top_k)
        results = [
            (
                index,
                1.0 - index / self.size,
                {
                    "id": index,
                    "title": "other namespace",
                    "content": "unrelated",
                    "knowledge_namespace": "private:other",
                    "record_type": "conversation_memory",
                },
            )
            for index in range(min(top_k, 1200))
        ]
        if top_k >= 1201:
            results.append(
                (
                    1200,
                    0.8,
                    {
                        "id": "sparse-hit",
                        "title": "sparse namespace result",
                        "content": "target",
                        "knowledge_namespace": "private:target",
                        "record_type": "conversation_memory",
                    },
                )
            )
        return results


@pytest.mark.asyncio
async def test_hybrid_search_filters_scope_before_ranking_and_fuses_fts(tmp_path):
    database = FakeDatabase()
    service = VectorService(database, vector_store_path=str(tmp_path))
    service.embedding_model = FakeEmbedding()
    service.vector_store = FakeStore()
    service._initialized = True

    results = await service.search_knowledge(
        "WAL visibility",
        top_k=3,
        threshold=0.2,
        knowledge_namespaces=["global:technical", "private:1"],
        collections=None,
        hybrid_search=True,
    )

    assert service.vector_store.search_sizes == [service.vector_store.size]
    assert all(item.get("knowledge_namespace") != "private:2" for item in results)
    assert {item.get("chunk_id") for item in results} >= {10, 11}
    assert any(item.get("id") == 101 for item in results)
    assert database.calls == [
        (
            "WAL visibility",
            {
                "limit": 40,
                "collections": None,
                "knowledge_namespaces": ["global:technical", "private:1"],
            },
        )
    ]
    lexical = next(item for item in results if item.get("chunk_id") == 11)
    assert lexical["record_type"] == "document_chunk"
    assert lexical["lexical_rank"] == 1


@pytest.mark.asyncio
async def test_filtered_search_progressively_oversamples_with_a_hard_limit(tmp_path):
    service = VectorService(None, vector_store_path=str(tmp_path))
    service.embedding_model = FakeEmbedding()
    service.vector_store = LargeSparseStore()
    service._initialized = True

    results = await service.search_knowledge(
        "WAL visibility",
        top_k=1,
        threshold=0.2,
        knowledge_namespaces=["private:target"],
        hybrid_search=False,
    )

    assert [item["id"] for item in results] == ["sparse-hit"]
    assert service.vector_store.search_sizes == [160, 320, 640, 1280]
    assert max(service.vector_store.search_sizes) <= service.MAX_FILTER_SCAN
