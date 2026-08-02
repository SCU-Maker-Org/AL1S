"""
向量存储服务
- 提供统一的向量存储和检索接口
- 支持多种向量存储后端（FAISS、InMemory等）
- 支持多种嵌入模型（TF-IDF、SentenceTransformers、HuggingFace等）
"""

import asyncio
import fcntl
import json
import os
import pickle
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# On Apple Silicon, importing FAISS before PyTorch can load conflicting OpenMP
# runtimes and segfault during model construction (pytorch/pytorch#149201).
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers 未安装，Sentence Transformer 功能不可用")

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS 未安装，将使用内存向量存储")

# 嵌入模型
try:
    from sklearn.feature_extraction.text import TfidfVectorizer

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn 未安装，TF-IDF 功能不可用")

from ..config import config
from ..models import KnowledgeEntry


class _IndexGenerationChanged(RuntimeError):
    """磁盘索引已被另一个进程替换。"""


class EmbeddingModel:
    """嵌入模型接口"""

    DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

    def __init__(
        self,
        model_type: str = DEFAULT_MODEL_NAME,
        model_name: str = None,
        revision: str = None,
        device: str = "cpu",
        batch_size: int = 8,
    ):
        self.model_type = model_type
        self.model_name = self._resolve_model_name(model_type, model_name)
        self.model_revision = revision
        self.device = device
        self.batch_size = batch_size
        self.dimension = 0
        self.model = None
        self._fitted = False
        self.normalized = True
        self.query_prompt_name = None
        self.query_prompt = None

        if model_type == "tfidf":
            self._init_tfidf_model()
        else:
            self._init_sentence_transformer_model(self.model_name)

    @classmethod
    def _resolve_model_name(cls, model_type: str, model_name: str = None) -> str:
        if model_name:
            return model_name
        if model_type == "sentence-transformers":
            return cls.DEFAULT_MODEL_NAME
        prefix = "sentence-transformers/"
        if model_type.startswith(prefix):
            resolved = model_type[len(prefix) :]
            return resolved or cls.DEFAULT_MODEL_NAME
        return model_type

    def _init_tfidf_model(self):
        """初始化TF-IDF模型"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn 未安装，无法使用 TF-IDF 模型")

        self.vectorizer = TfidfVectorizer(
            max_features=1024,
            stop_words=None,
            ngram_range=(1, 3),
            min_df=1,
            max_df=1.0,
            token_pattern=r"(?u)\b\w+\b",
            lowercase=True,
            sublinear_tf=True,
            norm="l2",
        )
        self.dimension = 1024
        logger.info("初始化 TF-IDF 嵌入模型")

    def _init_sentence_transformer_model(self, model_name: str):
        """初始化 Sentence Transformer 模型"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers 未安装，无法使用 Sentence Transformer 模型"
            )

        model_kwargs = {}
        if self.device != "auto":
            model_kwargs["device"] = self.device
        if self.model_revision:
            model_kwargs["revision"] = self.model_revision

        self.model = SentenceTransformer(model_name, **model_kwargs)
        if not self.model_revision:
            transformer_model = getattr(self.model, "transformers_model", None)
            model_config = getattr(transformer_model, "config", None)
            self.model_revision = getattr(model_config, "_commit_hash", None)
        self.dimension = self.model.get_sentence_embedding_dimension()
        prompts = getattr(self.model, "prompts", {}) or {}
        if "query" in prompts:
            self.query_prompt_name = "query"
            self.query_prompt = prompts["query"]
        logger.info(
            f"初始化 Sentence Transformer 模型: {model_name}, "
            f"维度: {self.dimension}, 设备: {self.device}"
        )

    async def fit(self, texts: List[str]):
        """训练模型（仅对TF-IDF有效）"""
        if self.model_type == "tfidf":
            await asyncio.to_thread(self.vectorizer.fit, texts)
            self._fitted = True
        else:
            self._fitted = True

    async def encode(
        self, texts: List[str], *, is_query: bool = False
    ) -> List[List[float]]:
        """编码文本列表"""
        if self.model_type == "tfidf":
            if not self._fitted:
                raise ValueError("TF-IDF 模型需要先训练")
            vectors = await asyncio.to_thread(self.vectorizer.transform, texts)
            vectors_array = vectors.toarray().astype("float32")
            if vectors_array.shape[1] < self.dimension:
                import numpy as np

                vectors_array = np.pad(
                    vectors_array,
                    ((0, 0), (0, self.dimension - vectors_array.shape[1])),
                )
            return vectors_array

        encode_kwargs = {
            "batch_size": self.batch_size,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        if is_query and self.query_prompt_name:
            encode_kwargs["prompt_name"] = self.query_prompt_name
        vectors = await asyncio.to_thread(self.model.encode, texts, **encode_kwargs)

        import numpy as np

        return np.asarray(vectors, dtype="float32")

    async def encode_single(self, text: str, *, is_query: bool = False) -> List[float]:
        """编码单个文本"""
        vectors = await self.encode([text], is_query=is_query)
        return vectors[0]


class VectorStore:
    """向量存储接口"""

    def __init__(self, backend: str = "faiss", dimension: int = 768):
        self.backend = backend
        self.dimension = dimension
        self.index = None
        self.metadata = {}  # 存储向量对应的元数据
        self._init_backend()

    def _init_backend(self):
        """初始化向量存储后端"""
        if self.backend == "faiss" and FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dimension)  # 内积相似度
            logger.info(f"初始化 FAISS 向量存储，维度: {self.dimension}")
        elif self.backend == "memory":
            self.vectors = []
            logger.info("初始化内存向量存储")
        else:
            raise ValueError(f"不支持的向量存储后端: {self.backend}")

    def reset(self):
        """重置为空存储。"""
        self.metadata = {}
        if self.backend == "faiss":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.backend == "memory":
            self.vectors = []

    @property
    def size(self) -> int:
        if self.backend == "faiss":
            return int(self.index.ntotal)
        return len(self.vectors)

    async def add_vectors(
        self, vectors: List[List[float]], metadata: List[Dict[str, Any]]
    ):
        """添加向量和元数据"""
        if self.backend == "faiss":
            import numpy as np

            vectors_array = np.array(vectors, dtype="float32")
            current_size = self.index.ntotal
            await asyncio.to_thread(self.index.add, vectors_array)

            # 保存元数据
            for i, meta in enumerate(metadata):
                self.metadata[current_size + i] = meta
        elif self.backend == "memory":
            start_idx = len(self.vectors)
            self.vectors.extend(vectors)
            for i, meta in enumerate(metadata):
                self.metadata[start_idx + i] = meta

        logger.debug(f"添加了 {len(vectors)} 个向量")

    async def search(
        self, query_vector: List[float], top_k: int = 5, threshold: float = 0.0
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """搜索相似向量"""
        if self.backend == "faiss":
            import numpy as np

            query_array = np.array([query_vector], dtype="float32")
            scores, indices = await asyncio.to_thread(
                self.index.search, query_array, top_k
            )

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score > threshold and idx in self.metadata:
                    results.append((idx, float(score), self.metadata[idx]))
            return results

        elif self.backend == "memory":
            import numpy as np

            if not self.vectors:
                return []

            # 计算相似度
            vectors_array = np.array(self.vectors)
            query_array = np.array(query_vector)
            similarities = np.dot(vectors_array, query_array)

            # 只排序候选集合，避免大内存索引为很小的 top_k 做全量排序。
            candidate_count = min(max(top_k, 0), len(similarities))
            if candidate_count == 0:
                return []
            if candidate_count == len(similarities):
                top_indices = np.argsort(similarities)[::-1]
            else:
                candidates = np.argpartition(similarities, -candidate_count)[
                    -candidate_count:
                ]
                top_indices = candidates[np.argsort(similarities[candidates])[::-1]]

            results = []
            for idx in top_indices:
                score = similarities[idx]
                if score > threshold:
                    results.append((int(idx), float(score), self.metadata.get(idx, {})))
            return results

        return []

    def save(self, file_path: str) -> bool:
        """以同目录临时文件保存，避免读者观察到半写入的单个文件。"""
        temporary_paths: List[Path] = []
        try:
            destination = Path(file_path)
            token = f"{os.getpid()}-{uuid.uuid4().hex}"
            if self.backend == "faiss":
                vector_path = destination.with_suffix(".faiss")
                vector_temp = vector_path.with_name(f".{vector_path.name}.{token}.tmp")
                temporary_paths.append(vector_temp)
                faiss.write_index(self.index, str(vector_temp))
                os.replace(vector_temp, vector_path)
            elif self.backend == "memory":
                vector_path = destination.with_suffix(".pkl")
                vector_temp = vector_path.with_name(f".{vector_path.name}.{token}.tmp")
                temporary_paths.append(vector_temp)
                with open(vector_temp, "wb") as f:
                    pickle.dump(self.vectors, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(vector_temp, vector_path)

            # 保存元数据
            metadata_path = Path(f"{file_path}_metadata.json")
            metadata_temp = metadata_path.with_name(
                f".{metadata_path.name}.{token}.tmp"
            )
            temporary_paths.append(metadata_temp)
            with open(metadata_temp, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(metadata_temp, metadata_path)

            logger.info(f"向量存储已保存到: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存向量存储失败: {e}")
            return False
        finally:
            for temporary_path in temporary_paths:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def load(self, file_path: str) -> bool:
        """加载向量存储"""
        try:
            loaded_index = None
            loaded_vectors = None
            if self.backend == "faiss":
                index_path = Path(f"{file_path}.faiss")
                if not index_path.exists():
                    return False
                loaded_index = faiss.read_index(str(index_path))
                if loaded_index.d != self.dimension:
                    logger.warning(
                        f"FAISS 索引维度不匹配: 现有 {loaded_index.d}, "
                        f"期望 {self.dimension}"
                    )
                    self.reset()
                    return False
            elif self.backend == "memory":
                vector_path = Path(f"{file_path}.pkl")
                if not vector_path.exists():
                    return False
                with open(vector_path, "rb") as f:
                    loaded_vectors = pickle.load(f)
                if any(len(vector) != self.dimension for vector in loaded_vectors):
                    logger.warning("内存向量维度与当前模型不匹配")
                    self.reset()
                    return False

            metadata_path = Path(f"{file_path}_metadata.json")
            if not metadata_path.exists():
                self.reset()
                return False
            with open(metadata_path, "r", encoding="utf-8") as f:
                raw_metadata = json.load(f)
            loaded_metadata = {int(k): v for k, v in raw_metadata.items()}

            vector_count = (
                loaded_index.ntotal if self.backend == "faiss" else len(loaded_vectors)
            )
            if set(loaded_metadata) != set(range(vector_count)):
                logger.warning(
                    f"向量与元数据数量或索引不一致: "
                    f"vectors={vector_count}, metadata={len(loaded_metadata)}"
                )
                self.reset()
                return False

            if self.backend == "faiss":
                self.index = loaded_index
            else:
                self.vectors = loaded_vectors
            self.metadata = loaded_metadata

            logger.info(f"向量存储已从 {file_path} 加载")
            return True
        except Exception as e:
            self.reset()
            logger.error(f"加载向量存储失败: {e}")
            return False


class VectorService:
    """向量存储服务"""

    INDEX_SCHEMA_VERSION = 5
    MAX_FILTER_SCAN = 4096
    REBUILD_SAVE_ATTEMPTS = 3

    def __init__(
        self, database_service=None, vector_store_path: str = "data/vector_store"
    ):
        self.database_service = database_service
        self.vector_store_path = Path(vector_store_path)
        self.vector_store_path.mkdir(exist_ok=True, parents=True)

        # 初始化组件
        self.embedding_model = None
        self.vector_store = None
        self._initialized = False
        self._index_lock = asyncio.Lock()
        self._loaded_generation: Optional[str] = None

    async def initialize(
        self,
        embedding_model_type: Optional[str] = None,
        vector_store_backend: str = "faiss",
        embedding_revision: Optional[str] = None,
        embedding_device: str = "cpu",
        embedding_batch_size: int = 8,
        force_rebuild: bool = False,
    ) -> bool:
        """初始化向量服务"""
        try:
            # 初始化嵌入模型 - 优先使用传入参数，否则从配置读取
            if not embedding_model_type:
                # 尝试从新的配置结构读取
                if hasattr(config, "agent") and hasattr(
                    config.agent, "embedding_model"
                ):
                    embedding_model_type = config.agent.embedding_model
                # 回退到旧的配置结构
                elif hasattr(config, "rag") and hasattr(config.rag, "embedding_model"):
                    embedding_model_type = config.rag.embedding_model
                else:
                    embedding_model_type = EmbeddingModel.DEFAULT_MODEL_NAME

            logger.info(f"使用嵌入模型: {embedding_model_type}")
            self.embedding_model = EmbeddingModel(
                embedding_model_type,
                revision=embedding_revision,
                device=embedding_device,
                batch_size=embedding_batch_size,
            )

            # 初始化向量存储
            self.vector_store = VectorStore(
                backend=vector_store_backend, dimension=self.embedding_model.dimension
            )

            # 摄取 CLI 可直接从 SQLite 构建一次，避免先加载/重建旧索引后再重复编码。
            initialized_data = (
                await self._rebuild_from_database()
                if force_rebuild
                else await self._load_existing_data()
            )
            if not initialized_data:
                return False

            self._initialized = True
            logger.info("向量服务初始化完成")
            return True

        except Exception as e:
            logger.error(f"向量服务初始化失败: {e}")
            return False

    def _index_manifest(self, generation: Optional[str] = None) -> Dict[str, Any]:
        return {
            "schema_version": self.INDEX_SCHEMA_VERSION,
            "backend": self.vector_store.backend,
            "model_id": self.embedding_model.model_name,
            "model_revision": self.embedding_model.model_revision,
            "dimension": self.embedding_model.dimension,
            "normalized": self.embedding_model.normalized,
            "query_prompt": self.embedding_model.query_prompt_name,
            "query_prompt_text": self.embedding_model.query_prompt,
            "generation": (
                self._loaded_generation if generation is None else generation
            ),
        }

    @property
    def _manifest_path(self) -> Path:
        return self.vector_store_path / "vector_store_manifest.json"

    @property
    def _process_lock_path(self) -> Path:
        return self.vector_store_path / ".vector_store.lock"

    @contextmanager
    def _process_index_lock(self, *, exclusive: bool):
        """序列化跨进程的索引文件读取与提交。"""
        self.vector_store_path.mkdir(exist_ok=True, parents=True)
        descriptor = os.open(self._process_lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "a+b") as lock_file:
            operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(lock_file.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _read_manifest_unlocked(self) -> Optional[Dict[str, Any]]:
        try:
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            return manifest if isinstance(manifest, dict) else None
        except (FileNotFoundError, OSError, ValueError, TypeError):
            return None

    @staticmethod
    def _manifest_generation(manifest: Optional[Dict[str, Any]]) -> Optional[str]:
        if not manifest:
            return None
        generation = manifest.get("generation")
        return generation if isinstance(generation, str) and generation else None

    def _manifest_model_matches(self, manifest: Optional[Dict[str, Any]]) -> bool:
        if not manifest or not self._manifest_generation(manifest):
            return False
        expected = self._index_manifest(generation=manifest["generation"])
        return manifest == expected

    def _manifest_matches(self) -> bool:
        if self.embedding_model.model_type == "tfidf":
            return False
        with self._process_index_lock(exclusive=False):
            return self._manifest_model_matches(self._read_manifest_unlocked())

    def _read_disk_generation(self) -> Optional[str]:
        with self._process_index_lock(exclusive=False):
            return self._manifest_generation(self._read_manifest_unlocked())

    def _load_compatible_store(self) -> bool:
        with self._process_index_lock(exclusive=False):
            manifest = self._read_manifest_unlocked()
            if not self._manifest_model_matches(manifest):
                return False
            generation = self._manifest_generation(manifest)
            if not self.vector_store.load(str(self._snapshot_store_file(generation))):
                return False
            self._loaded_generation = generation
            return True

    def _snapshot_store_file(self, generation: Optional[str]) -> Path:
        if generation:
            return self.vector_store_path / f"vector_store-{generation}"
        return self.vector_store_path / "vector_store"

    @staticmethod
    def _store_artifact_paths(store_file: Path) -> Tuple[Path, Path, Path]:
        return (
            store_file.with_suffix(".faiss"),
            store_file.with_suffix(".pkl"),
            Path(f"{store_file}_metadata.json"),
        )

    def _remove_store_artifacts(self, store_file: Path) -> None:
        for path in self._store_artifact_paths(store_file):
            try:
                path.unlink(missing_ok=True)
            except OSError as e:
                logger.warning("清理旧向量索引文件失败 {}: {}", path, e)

    def _remove_obsolete_snapshot(self, previous_generation: Optional[str]) -> None:
        if previous_generation:
            self._remove_store_artifacts(self._snapshot_store_file(previous_generation))
        self._remove_store_artifacts(self._snapshot_store_file(None))

    def _save_vector_store(self, expected_generation: Optional[str]) -> bool:
        """CAS 提交索引；磁盘版本变化时拒绝旧进程覆盖。"""
        with self._process_index_lock(exclusive=True):
            disk_generation = self._manifest_generation(self._read_manifest_unlocked())
            if disk_generation != expected_generation:
                raise _IndexGenerationChanged(
                    f"expected={expected_generation!r}, actual={disk_generation!r}"
                )

            generation = uuid.uuid4().hex
            store_file = self._snapshot_store_file(generation)
            if not self.vector_store.save(str(store_file)):
                self._remove_store_artifacts(store_file)
                return False

            token = f"{os.getpid()}-{uuid.uuid4().hex}"
            manifest_temp = self._manifest_path.with_name(
                f".{self._manifest_path.name}.{token}.tmp"
            )
            try:
                with open(manifest_temp, "w", encoding="utf-8") as f:
                    json.dump(
                        self._index_manifest(generation=generation),
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(manifest_temp, self._manifest_path)
                for path in (
                    *self._store_artifact_paths(store_file),
                    self._manifest_path,
                    self._process_lock_path,
                ):
                    if path.exists():
                        os.chmod(path, 0o600)
                self._loaded_generation = generation
                self._remove_obsolete_snapshot(disk_generation)
                return True
            except OSError as e:
                logger.error(f"保存向量索引清单失败: {e}")
                self._remove_store_artifacts(store_file)
                return False
            finally:
                try:
                    manifest_temp.unlink(missing_ok=True)
                except OSError:
                    pass

    async def _load_existing_data(self) -> bool:
        """加载现有的向量数据"""
        try:
            # 尝试从文件加载
            store_file = self.vector_store_path / "vector_store"
            if (
                self._manifest_path.exists()
                or store_file.with_suffix(".faiss").exists()
                or store_file.with_suffix(".pkl").exists()
            ):
                if await asyncio.to_thread(self._load_compatible_store):
                    await self._hydrate_knowledge_namespaces()
                    logger.info("已加载兼容的向量存储")
                    return True
                logger.warning("向量索引与当前模型不兼容，将从数据库重建")

            # 如果没有现有文件，从数据库重建
            if not self.database_service:
                logger.error("没有数据库服务，无法重建不兼容的向量索引")
                return False

            logger.info("正在从数据库重建向量存储...")
            return await self._rebuild_from_database()

        except Exception as e:
            logger.error(f"加载现有数据失败: {e}")
            return False

    async def _hydrate_knowledge_namespaces(self) -> None:
        """为升级前的向量元数据补齐知识命名空间。"""
        if not self.database_service or not self.vector_store:
            return
        entries = await asyncio.to_thread(
            self.database_service.get_all_knowledge_entries
        )
        namespaces = {
            entry.get("id"): entry.get("knowledge_namespace", "") for entry in entries
        }
        changed = False
        for metadata in self.vector_store.metadata.values():
            if not metadata.get("knowledge_namespace"):
                metadata["knowledge_namespace"] = namespaces.get(metadata.get("id"), "")
                changed = True
        if changed:
            try:
                saved = await asyncio.to_thread(
                    self._save_vector_store, self._loaded_generation
                )
            except _IndexGenerationChanged:
                logger.info("补齐命名空间时索引已更新，改为从数据库重建")
                await self._rebuild_from_database()
            else:
                if not saved:
                    logger.error("保存补齐命名空间后的向量索引失败")

    async def _rebuild_from_database(self) -> bool:
        """从数据库重建向量存储"""
        for attempt in range(1, self.REBUILD_SAVE_ATTEMPTS + 1):
            try:
                expected_generation = await asyncio.to_thread(
                    self._read_disk_generation
                )

                # 获取所有知识条目
                knowledge_entries = await asyncio.to_thread(
                    self.database_service.get_all_knowledge_entries
                )
                rag_chunks = []
                if hasattr(self.database_service, "get_all_rag_chunks"):
                    rag_chunks = await asyncio.to_thread(
                        self.database_service.get_all_rag_chunks
                    )

                # 准备文本和元数据
                texts = []
                metadata = []

                for entry in knowledge_entries:
                    content = (
                        f"{entry.get('title', '')} {entry.get('content', '')} "
                        f"{entry.get('summary', '')}"
                    )
                    texts.append(content.strip())
                    metadata.append(
                        {
                            "id": entry.get("id"),
                            "title": entry.get("title", ""),
                            "content": entry.get("content", ""),
                            "summary": entry.get("summary", ""),
                            "keywords": entry.get("keywords", ""),
                            "category": entry.get("category", "general"),
                            "importance_score": entry.get("importance_score", 0.0),
                            "knowledge_namespace": entry.get("knowledge_namespace", ""),
                            "record_type": "conversation_memory",
                        }
                    )

                for chunk in rag_chunks:
                    heading = chunk.get("heading_path", "")
                    title = chunk.get("title", "")
                    content = chunk.get("content", "")
                    texts.append(f"{title} {heading} {content}".strip())
                    metadata.append(
                        {
                            "id": f"rag:{chunk.get('chunk_id')}",
                            "chunk_id": chunk.get("chunk_id"),
                            "document_id": chunk.get("document_id"),
                            "title": title,
                            "content": content,
                            "summary": "",
                            "keywords": " ".join(
                                value
                                for value in (
                                    chunk.get("domain", ""),
                                    chunk.get("subdomain", ""),
                                    chunk.get("product", ""),
                                )
                                if value
                            ),
                            "category": chunk.get("domain", ""),
                            "importance_score": min(
                                max(
                                    float(chunk.get("trust_level", 50)) / 100.0,
                                    0.0,
                                ),
                                1.0,
                            ),
                            "knowledge_namespace": chunk.get("knowledge_namespace", ""),
                            "collection": chunk.get("collection", ""),
                            "source_uri": chunk.get("source_uri", ""),
                            "heading_path": heading,
                            "version": chunk.get("version", ""),
                            "product": chunk.get("product", ""),
                            "license": chunk.get("license", ""),
                            "trust_level": chunk.get("trust_level", 50),
                            "record_type": "document_chunk",
                        }
                    )

                if texts:
                    await self.embedding_model.fit(texts)
                    vectors = await self.embedding_model.encode(texts)
                else:
                    vectors = []

                # 编码成功后再替换内存索引，避免模型异常提前清空旧索引。
                self.vector_store.reset()
                if len(vectors):
                    await self.vector_store.add_vectors(vectors, metadata)

                try:
                    saved = await asyncio.to_thread(
                        self._save_vector_store, expected_generation
                    )
                except _IndexGenerationChanged:
                    logger.warning(
                        "重建提交时发现索引 generation 已变化，第 {}/{} 次重试",
                        attempt,
                        self.REBUILD_SAVE_ATTEMPTS,
                    )
                    continue
                if not saved:
                    return False

                if not texts:
                    logger.info("数据库中没有知识条目或文档分块")
                else:
                    logger.info(
                        "从数据库重建了 {} 个向量（聊天知识 {}，文档分块 {}）",
                        len(texts),
                        len(knowledge_entries),
                        len(rag_chunks),
                    )
                return True
            except Exception as e:
                logger.error(f"从数据库重建向量存储失败: {e}")
                return False

        logger.error("索引持续被其他进程更新，重建提交已放弃")
        return False

    async def add_knowledge(self, knowledge_entry: KnowledgeEntry) -> bool:
        """添加知识条目"""
        try:
            if not self._initialized:
                logger.warning("向量服务未初始化")
                return False

            # 组合文本内容
            content = f"{knowledge_entry.title} {knowledge_entry.content} {knowledge_entry.summary}"

            # TF-IDF 的词表依赖完整语料；数据库已先写入新条目，因此整体重建。
            if self.embedding_model.model_type == "tfidf":
                if not self.database_service:
                    logger.error("TF-IDF 添加知识需要数据库服务以重建词表")
                    return False
                async with self._index_lock:
                    return await self._rebuild_from_database()

            async with self._index_lock:
                disk_generation = await asyncio.to_thread(self._read_disk_generation)
                if disk_generation != self._loaded_generation:
                    logger.info("检测到外部索引更新，增量写入前先从数据库重建")
                    return await self._rebuild_from_database()

                # 生成向量
                vector = await self.embedding_model.encode_single(content.strip())

                # 添加到向量存储
                item = knowledge_entry.to_dict()
                item["record_type"] = "conversation_memory"
                await self.vector_store.add_vectors([vector], [item])

                try:
                    saved = await asyncio.to_thread(
                        self._save_vector_store, self._loaded_generation
                    )
                except _IndexGenerationChanged:
                    logger.info("增量编码期间索引被外部重建，改为从数据库重建")
                    return await self._rebuild_from_database()
                if not saved:
                    return False

            logger.debug(f"添加知识条目: {knowledge_entry.title}")
            return True

        except Exception as e:
            logger.error(f"添加知识条目失败: {e}")
            return False

    async def search_knowledge(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.1,
        knowledge_namespace: Optional[str] = None,
        knowledge_namespaces: Optional[List[str]] = None,
        collections: Optional[List[str]] = None,
        hybrid_search: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """融合稠密向量与 FTS5 排名，并在 Top-K 前执行作用域过滤。"""
        try:
            if not self._initialized:
                logger.warning("向量服务未初始化")
                return []

            # 生成查询向量
            query_vector = await self.embedding_model.encode_single(
                query, is_query=True
            )

            namespaces = list(knowledge_namespaces or [])
            if (
                knowledge_namespace is not None
                and knowledge_namespace not in namespaces
            ):
                namespaces.append(knowledge_namespace)

            candidate_k = max(top_k, int(config.rag.candidate_k))
            async with self._index_lock:
                store_size = self.vector_store.size
                filtered_search = bool(namespaces or collections)
                max_scan = min(store_size, self.MAX_FILTER_SCAN)
                if filtered_search:
                    search_size = min(
                        max_scan,
                        max(candidate_k * 4, candidate_k),
                    )
                else:
                    search_size = min(store_size, candidate_k)

                dense_results = []
                while search_size > 0:
                    results = await self.vector_store.search(
                        query_vector, search_size, threshold
                    )
                    dense_results = []
                    for _, score, metadata in results:
                        if (
                            namespaces
                            and metadata.get("knowledge_namespace", "")
                            not in namespaces
                        ):
                            continue
                        if (
                            collections
                            and metadata.get("collection", "") not in collections
                        ):
                            continue
                        item = metadata.copy()
                        item["similarity_score"] = score
                        dense_results.append(item)
                        if len(dense_results) >= candidate_k:
                            break

                    if (
                        not filtered_search
                        or len(dense_results) >= top_k
                        or search_size >= max_scan
                    ):
                        break
                    search_size = min(max_scan, search_size * 2)

            use_hybrid = (
                config.rag.hybrid_search if hybrid_search is None else hybrid_search
            )
            lexical_results = []
            if (
                use_hybrid
                and self.database_service
                and hasattr(self.database_service, "search_rag_chunks_lexical")
            ):
                lexical_results = await asyncio.to_thread(
                    self.database_service.search_rag_chunks_lexical,
                    query,
                    limit=candidate_k,
                    collections=collections,
                    knowledge_namespaces=namespaces or None,
                )

            rrf_k = float(config.rag.rrf_k)
            fused: Dict[str, Dict[str, Any]] = {}

            def result_key(item: Dict[str, Any]) -> str:
                chunk_id = item.get("chunk_id")
                if chunk_id is not None:
                    return f"document_chunk:{chunk_id}"
                return f"conversation_memory:{item.get('id')}"

            for rank, item in enumerate(dense_results, 1):
                key = result_key(item)
                fused[key] = dict(item)
                fused[key]["retrieval_score"] = 1.0 / (rrf_k + rank)
                fused[key]["dense_rank"] = rank

            for rank, item in enumerate(lexical_results, 1):
                key = result_key(item)
                if key not in fused:
                    fused[key] = {
                        **item,
                        "id": f"rag:{item.get('chunk_id')}",
                        "summary": "",
                        "keywords": " ".join(
                            value
                            for value in (
                                item.get("domain", ""),
                                item.get("subdomain", ""),
                                item.get("product", ""),
                            )
                            if value
                        ),
                        "category": item.get("domain", ""),
                        "record_type": "document_chunk",
                        "similarity_score": 0.0,
                        "retrieval_score": 0.0,
                    }
                fused[key]["retrieval_score"] += 1.0 / (rrf_k + rank)
                fused[key]["lexical_rank"] = rank

            knowledge_results = sorted(
                fused.values(),
                key=lambda item: (
                    float(item.get("retrieval_score", 0.0)),
                    float(item.get("similarity_score", 0.0)),
                    float(item.get("trust_level", 0.0)),
                ),
                reverse=True,
            )[:top_k]

            logger.debug(f"搜索到 {len(knowledge_results)} 个相关知识")
            return knowledge_results

        except Exception as e:
            logger.error(f"搜索知识失败: {e}")
            return []

    async def rebuild(self) -> bool:
        """串行化地从 SQLite 重建完整稠密索引。"""
        if not self.database_service or not self.vector_store:
            return False
        async with self._index_lock:
            return await self._rebuild_from_database()

    def cleanup(self):
        """清理资源"""
        try:
            if self.embedding_model:
                del self.embedding_model
                self.embedding_model = None

            if self.vector_store:
                del self.vector_store
                self.vector_store = None

            logger.debug("向量服务资源清理完成")
        except Exception as e:
            logger.debug(f"向量服务资源清理失败: {e}")

    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except Exception:
            pass
