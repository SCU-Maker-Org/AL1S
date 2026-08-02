"""
向量存储服务
- 提供统一的向量存储和检索接口
- 支持多种向量存储后端（FAISS、InMemory等）
- 支持多种嵌入模型（TF-IDF、SentenceTransformers、HuggingFace等）
"""

import asyncio
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 向量存储后端
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

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers 未安装，Sentence Transformer 功能不可用")

from ..config import config
from ..models import KnowledgeEntry


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

            # 获取top_k结果
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                score = similarities[idx]
                if score > threshold:
                    results.append((int(idx), float(score), self.metadata.get(idx, {})))
            return results

        return []

    def save(self, file_path: str) -> bool:
        """保存向量存储"""
        try:
            if self.backend == "faiss":
                faiss.write_index(self.index, f"{file_path}.faiss")
            elif self.backend == "memory":
                with open(f"{file_path}.pkl", "wb") as f:
                    pickle.dump(self.vectors, f)

            # 保存元数据
            with open(f"{file_path}_metadata.json", "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"向量存储已保存到: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存向量存储失败: {e}")
            return False

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

    INDEX_SCHEMA_VERSION = 3

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

    async def initialize(
        self,
        embedding_model_type: Optional[str] = None,
        vector_store_backend: str = "faiss",
        embedding_revision: Optional[str] = None,
        embedding_device: str = "cpu",
        embedding_batch_size: int = 8,
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

            # 加载现有数据
            if not await self._load_existing_data():
                return False

            self._initialized = True
            logger.info("向量服务初始化完成")
            return True

        except Exception as e:
            logger.error(f"向量服务初始化失败: {e}")
            return False

    def _index_manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": self.INDEX_SCHEMA_VERSION,
            "backend": self.vector_store.backend,
            "model_id": self.embedding_model.model_name,
            "model_revision": self.embedding_model.model_revision,
            "dimension": self.embedding_model.dimension,
            "normalized": self.embedding_model.normalized,
            "query_prompt": self.embedding_model.query_prompt_name,
            "query_prompt_text": self.embedding_model.query_prompt,
        }

    @property
    def _manifest_path(self) -> Path:
        return self.vector_store_path / "vector_store_manifest.json"

    def _manifest_matches(self) -> bool:
        if self.embedding_model.model_type == "tfidf":
            return False
        try:
            if not self._manifest_path.exists():
                return False
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            return existing == self._index_manifest()
        except (OSError, ValueError, TypeError):
            return False

    def _save_vector_store(self) -> bool:
        store_file = self.vector_store_path / "vector_store"
        if not self.vector_store.save(str(store_file)):
            return False
        manifest_temp = self._manifest_path.with_suffix(".json.tmp")
        try:
            with open(manifest_temp, "w", encoding="utf-8") as f:
                json.dump(self._index_manifest(), f, ensure_ascii=False, indent=2)
            manifest_temp.replace(self._manifest_path)
            return True
        except OSError as e:
            logger.error(f"保存向量索引清单失败: {e}")
            return False

    async def _load_existing_data(self) -> bool:
        """加载现有的向量数据"""
        try:
            # 尝试从文件加载
            store_file = self.vector_store_path / "vector_store"
            if (
                store_file.with_suffix(".faiss").exists()
                or store_file.with_suffix(".pkl").exists()
            ):
                if self._manifest_matches() and self.vector_store.load(str(store_file)):
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
            self._save_vector_store()

    async def _rebuild_from_database(self) -> bool:
        """从数据库重建向量存储"""
        try:
            # 获取所有知识条目
            knowledge_entries = await asyncio.to_thread(
                self.database_service.get_all_knowledge_entries
            )

            self.vector_store.reset()

            if not knowledge_entries:
                logger.info("数据库中没有知识条目")
                return self._save_vector_store()

            # 准备文本和元数据
            texts = []
            metadata = []

            for entry in knowledge_entries:
                # 组合文本内容
                content = f"{entry.get('title', '')} {entry.get('content', '')} {entry.get('summary', '')}"
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
                    }
                )

            # 训练嵌入模型
            await self.embedding_model.fit(texts)

            # 生成向量
            vectors = await self.embedding_model.encode(texts)

            # 添加到向量存储
            await self.vector_store.add_vectors(vectors, metadata)

            # 保存到文件
            if not self._save_vector_store():
                return False

            logger.info(f"从数据库重建了 {len(texts)} 个向量")
            return True

        except Exception as e:
            logger.error(f"从数据库重建向量存储失败: {e}")
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
                return await self._rebuild_from_database()

            # 生成向量
            vector = await self.embedding_model.encode_single(content.strip())

            # 添加到向量存储
            metadata = [knowledge_entry.to_dict()]
            await self.vector_store.add_vectors([vector], metadata)

            # 保存到文件
            if not self._save_vector_store():
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
    ) -> List[Dict[str, Any]]:
        """搜索相关知识"""
        try:
            if not self._initialized:
                logger.warning("向量服务未初始化")
                return []

            # 生成查询向量
            query_vector = await self.embedding_model.encode_single(
                query, is_query=True
            )

            # 搜索相似向量
            search_size = (
                top_k if knowledge_namespace is None else max(top_k * 10, top_k)
            )
            results = await self.vector_store.search(
                query_vector, search_size, threshold
            )

            # 格式化结果
            knowledge_results = []
            for idx, score, metadata in results:
                if (
                    knowledge_namespace is not None
                    and metadata.get("knowledge_namespace", "") != knowledge_namespace
                ):
                    continue
                result = metadata.copy()
                result["similarity_score"] = score
                knowledge_results.append(result)
                if len(knowledge_results) >= top_k:
                    break

            logger.debug(f"搜索到 {len(knowledge_results)} 个相关知识")
            return knowledge_results

        except Exception as e:
            logger.error(f"搜索知识失败: {e}")
            return []

    def cleanup(self):
        """清理资源"""
        try:
            if self.vector_store and self._initialized:
                # 保存当前状态
                self._save_vector_store()

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
