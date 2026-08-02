from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest

from src.infra import vector as vector_module
from src.infra.vector import EmbeddingModel, VectorService, VectorStore
from src.models import KnowledgeEntry


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


class MutableDatabase:
    def __init__(self, entries):
        self.entries = entries

    def get_all_knowledge_entries(self):
        return deepcopy(self.entries)


class SmallEmbeddingModel:
    model_type = "test-embedding"
    model_name = "test-embedding"
    model_revision = "v1"
    dimension = 2
    normalized = True
    query_prompt_name = None
    query_prompt = None

    def __init__(self, encode_single_hook=None):
        self.encode_single_hook = encode_single_hook

    async def fit(self, _texts):
        return None

    async def encode(self, texts):
        return np.asarray(
            [[1.0, float(index % 2)] for index, _ in enumerate(texts)],
            dtype=np.float32,
        )

    async def encode_single(self, _text, *, is_query=False):
        assert is_query is False
        if self.encode_single_hook:
            hook = self.encode_single_hook
            self.encode_single_hook = None
            await hook()
        return np.asarray([1.0, 0.0], dtype=np.float32)


def make_memory_service(database, path, embedding=None):
    service = VectorService(database, vector_store_path=str(path))
    service.embedding_model = embedding or SmallEmbeddingModel()
    service.vector_store = VectorStore(backend="memory", dimension=2)
    return service


def entry(entry_id, title):
    return {
        "id": entry_id,
        "title": title,
        "content": f"{title} content",
        "summary": "summary",
        "keywords": "test",
        "category": "test",
        "importance_score": 0.5,
        "knowledge_namespace": "global",
    }


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
async def test_stale_service_rebuilds_instead_of_overwriting_new_generation(tmp_path):
    database = MutableDatabase([entry(1, "initial")])
    stale_bot = make_memory_service(database, tmp_path)
    assert await stale_bot._rebuild_from_database() is True
    stale_bot._initialized = True
    original_generation = stale_bot._loaded_generation

    cli = make_memory_service(database, tmp_path)
    assert await cli._load_existing_data() is True
    database.entries.append(entry(2, "cli document"))
    assert await cli.rebuild() is True
    cli_generation = cli._loaded_generation
    assert cli_generation != original_generation

    database.entries.append(entry(3, "new bot memory"))
    new_entry = KnowledgeEntry(
        id=3,
        title="new bot memory",
        content="new bot memory content",
        summary="summary",
        knowledge_namespace="global",
    )
    assert await stale_bot.add_knowledge(new_entry) is True

    verifier = make_memory_service(database, tmp_path)
    assert await verifier._load_existing_data() is True
    assert {item["id"] for item in verifier.vector_store.metadata.values()} == {
        1,
        2,
        3,
    }
    assert verifier._loaded_generation not in {
        original_generation,
        cli_generation,
    }


@pytest.mark.asyncio
async def test_generation_change_during_incremental_encode_retries_from_database(
    tmp_path,
):
    database = MutableDatabase([entry(1, "initial")])
    writer = make_memory_service(database, tmp_path)
    assert await writer._rebuild_from_database() is True
    writer._initialized = True

    external_rebuilder = make_memory_service(database, tmp_path)
    assert await external_rebuilder._load_existing_data() is True
    database.entries.append(entry(2, "concurrent memory"))

    generation_before_race = writer._loaded_generation
    writer.embedding_model.encode_single_hook = external_rebuilder.rebuild
    new_entry = KnowledgeEntry(
        id=2,
        title="concurrent memory",
        content="concurrent memory content",
        summary="summary",
        knowledge_namespace="global",
    )

    assert await writer.add_knowledge(new_entry) is True
    assert external_rebuilder._loaded_generation != generation_before_race
    assert writer._loaded_generation != external_rebuilder._loaded_generation

    verifier = make_memory_service(database, tmp_path)
    assert await verifier._load_existing_data() is True
    assert {item["id"] for item in verifier.vector_store.metadata.values()} == {1, 2}


@pytest.mark.asyncio
async def test_tfidf_accepts_a_single_document():
    if not vector_module.SKLEARN_AVAILABLE:
        pytest.skip("scikit-learn is not installed")

    model = EmbeddingModel("tfidf")
    await model.fit(["only one knowledge entry"])
    vectors = await model.encode(["only one knowledge entry"])

    assert vectors.shape == (1, 1024)
    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0)
