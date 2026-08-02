from __future__ import annotations

import json

import numpy as np
import pytest

from src.infra import vector as vector_module
from src.infra.vector import EmbeddingModel, VectorService, VectorStore


class FakeSentenceTransformer:
    instances = []

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.init_kwargs = kwargs
        self.prompts = {"query": "Instruct: retrieve relevant passages"}
        self.encode_calls = []
        self.__class__.instances.append(self)

    def get_sentence_embedding_dimension(self):
        return 1024

    def encode(self, texts, **kwargs):
        self.encode_calls.append((texts, kwargs))
        return np.ones((len(texts), 1024), dtype=np.float32)


@pytest.fixture
def fake_sentence_transformer(monkeypatch):
    FakeSentenceTransformer.instances.clear()
    monkeypatch.setattr(vector_module, "SENTENCE_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(
        vector_module,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )
    return FakeSentenceTransformer


def test_sentence_transformer_model_id_is_not_replaced(
    fake_sentence_transformer,
):
    model = EmbeddingModel(
        "sentence-transformers/Qwen/Qwen3-Embedding-0.6B",
        revision="model-commit",
        device="cpu",
    )

    loaded = fake_sentence_transformer.instances[0]
    assert model.model_name == "Qwen/Qwen3-Embedding-0.6B"
    assert loaded.model_name == "Qwen/Qwen3-Embedding-0.6B"
    assert loaded.init_kwargs == {"device": "cpu", "revision": "model-commit"}
    assert model.model_revision == "model-commit"
    assert model.query_prompt == "Instruct: retrieve relevant passages"
    assert model.dimension == 1024


@pytest.mark.asyncio
async def test_query_encoding_uses_prompt_and_normalized_embeddings(
    fake_sentence_transformer,
):
    model = EmbeddingModel(
        "Qwen/Qwen3-Embedding-0.6B",
        device="cpu",
        batch_size=3,
    )

    vectors = await model.encode(["query text"], is_query=True)
    loaded = fake_sentence_transformer.instances[0]
    texts, kwargs = loaded.encode_calls[0]

    assert texts == ["query text"]
    assert kwargs == {
        "batch_size": 3,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False,
        "prompt_name": "query",
    }
    assert vectors.dtype == np.float32

    await model.encode(["document text"])
    _, document_kwargs = loaded.encode_calls[1]
    assert "prompt_name" not in document_kwargs
    assert document_kwargs["normalize_embeddings"] is True


@pytest.mark.asyncio
async def test_faiss_dimension_mismatch_resets_metadata(tmp_path):
    if not vector_module.FAISS_AVAILABLE:
        pytest.skip("FAISS is not installed")

    store_path = tmp_path / "vector_store"
    old_store = VectorStore(backend="faiss", dimension=3)
    await old_store.add_vectors([[1.0, 0.0, 0.0]], [{"id": 1}])
    assert old_store.save(str(store_path)) is True

    upgraded_store = VectorStore(backend="faiss", dimension=4)
    upgraded_store.metadata = {99: {"id": "stale"}}

    assert upgraded_store.load(str(store_path)) is False
    assert upgraded_store.metadata == {}
    assert upgraded_store.index.ntotal == 0
    assert upgraded_store.index.d == 4


class FakeEmbeddingModel:
    model_type = "Qwen/Qwen3-Embedding-0.6B"
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    model_revision = "model-commit"
    dimension = 1024
    normalized = True
    query_prompt_name = "query"
    query_prompt = "Instruct: retrieve relevant passages"

    def __init__(self):
        self.fit_calls = []
        self.encode_calls = []

    async def fit(self, texts):
        self.fit_calls.append(texts)

    async def encode(self, texts):
        self.encode_calls.append(texts)
        return np.ones((len(texts), self.dimension), dtype=np.float32)


class FakeVectorStore:
    backend = "memory"

    def __init__(self):
        self.metadata = {}
        self.load_calls = 0
        self.reset_calls = 0
        self.add_calls = []
        self.save_calls = 0

    def load(self, _file_path):
        self.load_calls += 1
        return True

    def reset(self):
        self.reset_calls += 1
        self.metadata = {}

    async def add_vectors(self, vectors, metadata):
        self.add_calls.append((vectors, metadata))
        self.metadata = {index: value for index, value in enumerate(metadata)}

    def save(self, _file_path):
        self.save_calls += 1
        return True


class FakeDatabase:
    def get_all_knowledge_entries(self):
        return [
            {
                "id": 7,
                "title": "Current knowledge",
                "content": "Rebuild this entry",
                "summary": "summary",
                "keywords": "upgrade",
                "category": "test",
                "importance_score": 0.8,
                "knowledge_namespace": "global",
            }
        ]


@pytest.mark.asyncio
async def test_model_manifest_mismatch_rebuilds_without_loading_old_index(
    tmp_path,
):
    service = VectorService(FakeDatabase(), vector_store_path=str(tmp_path))
    service.embedding_model = FakeEmbeddingModel()
    service.vector_store = FakeVectorStore()

    (tmp_path / "vector_store.pkl").write_bytes(b"old index marker")
    old_manifest = service._index_manifest() | {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2"
    }
    service._manifest_path.write_text(
        json.dumps(old_manifest),
        encoding="utf-8",
    )

    assert await service._load_existing_data() is True
    assert service.vector_store.load_calls == 0
    assert service.vector_store.reset_calls == 1
    assert len(service.vector_store.add_calls) == 1
    assert service.embedding_model.fit_calls == [
        ["Current knowledge Rebuild this entry summary"]
    ]
    assert service.embedding_model.encode_calls == [
        ["Current knowledge Rebuild this entry summary"]
    ]

    manifest_text = service._manifest_path.read_text(encoding="utf-8")
    updated_manifest = json.loads(manifest_text)
    assert updated_manifest == service._index_manifest()


@pytest.mark.asyncio
async def test_tfidf_accepts_a_single_document():
    if not vector_module.SKLEARN_AVAILABLE:
        pytest.skip("scikit-learn is not installed")

    model = EmbeddingModel("tfidf")
    await model.fit(["only one knowledge entry"])
    vectors = await model.encode(["only one knowledge entry"])

    assert vectors.shape == (1, 1024)
    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0)
