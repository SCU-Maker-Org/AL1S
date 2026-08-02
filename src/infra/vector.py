"""
向量存储服务
- 提供统一的向量存储和检索接口
- 支持多种向量存储后端（FAISS、InMemory等）
- 支持多种嵌入模型（TF-IDF、SentenceTransformers、HuggingFace等）
"""

import asyncio
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

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

    def __init__(self, model_type: str = "tfidf", model_name: str = None):
        self.model_type = model_type
        self.model_name = model_name or model_type
        self.dimension = 0
        self.model = None
        self._fitted = False

        if model_type == "tfidf":
            self._init_tfidf_model()
        elif model_type.startswith("sentence-transformers"):
            self._init_sentence_transformer_model(
                model_name or "paraphrase-multilingual-MiniLM-L12-v2"
            )
        else:
            raise ValueError(f"不支持的嵌入模型类型: {model_type}")

    def _init_tfidf_model(self):
        """初始化TF-IDF模型"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn 未安装，无法使用 TF-IDF 模型")

        self.vectorizer = TfidfVectorizer(
            max_features=1024,
            stop_words=None,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.8,
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

        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(
            f"初始化 Sentence Transformer 模型: {model_name}, 维度: {self.dimension}"
        )

    async def fit(self, texts: List[str]):
        """训练模型（仅对TF-IDF有效）"""
        if self.model_type == "tfidf":
            await asyncio.to_thread(self.vectorizer.fit, texts)
            self._fitted = True
        else:
            self._fitted = True

    async def encode(self, texts: List[str]) -> List[List[float]]:
        """编码文本列表"""
        if self.model_type == "tfidf":
            if not self._fitted:
                raise ValueError("TF-IDF 模型需要先训练")
            vectors = await asyncio.to_thread(self.vectorizer.transform, texts)
            return vectors.toarray().astype("float32")
        else:
            return await asyncio.to_thread(self.model.encode, texts)

    async def encode_single(self, text: str) -> List[float]:
        """编码单个文本"""
        vectors = await self.encode([text])
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

    def save(self, file_path: str):
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
        except Exception as e:
            logger.error(f"保存向量存储失败: {e}")

    def load(self, file_path: str):
        """加载向量存储"""
        try:
            if self.backend == "faiss":
                if Path(f"{file_path}.faiss").exists():
                    loaded_index = faiss.read_index(f"{file_path}.faiss")

                    # 检查维度兼容性
                    if loaded_index.d != self.dimension:
                        logger.warning(
                            f"FAISS 索引维度不匹配: 现有 {loaded_index.d}, 期望 {self.dimension}"
                        )
                        logger.warning("将重新初始化空索引")
                        # 保持原来的空索引，不加载不兼容的索引
                    else:
                        self.index = loaded_index
                        logger.info(f"FAISS 索引加载成功，维度: {self.index.d}")
            elif self.backend == "memory":
                if Path(f"{file_path}.pkl").exists():
                    with open(f"{file_path}.pkl", "rb") as f:
                        self.vectors = pickle.load(f)

            # 加载元数据（只有在索引兼容时才加载）
            if (
                self.backend == "faiss"
                and hasattr(self, "index")
                and self.index is not None
            ):
                metadata_file = f"{file_path}_metadata.json"
                if Path(metadata_file).exists():
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        # 转换键为整数
                        loaded_metadata = json.load(f)
                        self.metadata = {int(k): v for k, v in loaded_metadata.items()}
                        logger.info(f"加载了 {len(self.metadata)} 个元数据条目")
            elif self.backend == "memory":
                metadata_file = f"{file_path}_metadata.json"
                if Path(metadata_file).exists():
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        loaded_metadata = json.load(f)
                        self.metadata = {int(k): v for k, v in loaded_metadata.items()}

            logger.info(f"向量存储已从 {file_path} 加载")
        except Exception as e:
            logger.error(f"加载向量存储失败: {e}")


class VectorService:
    """向量存储服务"""

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
        self, embedding_model_type: str = "tfidf", vector_store_backend: str = "faiss"
    ) -> bool:
        """初始化向量服务"""
        try:
            # 初始化嵌入模型 - 优先使用传入参数，否则从配置读取
            if not embedding_model_type or embedding_model_type == "tfidf":
                # 尝试从新的配置结构读取
                if hasattr(config, "agent") and hasattr(
                    config.agent, "embedding_model"
                ):
                    embedding_model_type = config.agent.embedding_model
                # 回退到旧的配置结构
                elif hasattr(config, "rag") and hasattr(config.rag, "embedding_model"):
                    embedding_model_type = config.rag.embedding_model
                else:
                    embedding_model_type = "tfidf"  # 默认值

            logger.info(f"使用嵌入模型: {embedding_model_type}")
            self.embedding_model = EmbeddingModel(embedding_model_type)

            # 初始化向量存储
            self.vector_store = VectorStore(
                backend=vector_store_backend, dimension=self.embedding_model.dimension
            )

            # 加载现有数据
            await self._load_existing_data()

            self._initialized = True
            logger.info("向量服务初始化完成")
            return True

        except Exception as e:
            logger.error(f"向量服务初始化失败: {e}")
            return False

    async def _load_existing_data(self):
        """加载现有的向量数据"""
        try:
            # 尝试从文件加载
            store_file = self.vector_store_path / "vector_store"
            if (
                store_file.with_suffix(".faiss").exists()
                or store_file.with_suffix(".pkl").exists()
            ):
                self.vector_store.load(str(store_file))
                await self._hydrate_knowledge_namespaces()
                logger.info("已加载现有向量存储")
                return

            # 如果没有现有文件，从数据库重建
            if not self.database_service:
                logger.info("没有数据库服务，跳过数据加载")
                return

            logger.info("正在从数据库重建向量存储...")
            await self._rebuild_from_database()

        except Exception as e:
            logger.error(f"加载现有数据失败: {e}")

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
            store_file = self.vector_store_path / "vector_store"
            self.vector_store.save(str(store_file))

    async def _rebuild_from_database(self):
        """从数据库重建向量存储"""
        try:
            # 获取所有知识条目
            knowledge_entries = await asyncio.to_thread(
                self.database_service.get_all_knowledge_entries
            )

            if not knowledge_entries:
                logger.info("数据库中没有知识条目")
                return

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
            store_file = self.vector_store_path / "vector_store"
            self.vector_store.save(str(store_file))

            logger.info(f"从数据库重建了 {len(texts)} 个向量")

        except Exception as e:
            logger.error(f"从数据库重建向量存储失败: {e}")

    async def add_knowledge(self, knowledge_entry: KnowledgeEntry) -> bool:
        """添加知识条目"""
        try:
            if not self._initialized:
                logger.warning("向量服务未初始化")
                return False

            # 组合文本内容
            content = f"{knowledge_entry.title} {knowledge_entry.content} {knowledge_entry.summary}"

            # 生成向量
            vector = await self.embedding_model.encode_single(content.strip())

            # 添加到向量存储
            metadata = [knowledge_entry.to_dict()]
            await self.vector_store.add_vectors([vector], metadata)

            # 保存到文件
            store_file = self.vector_store_path / "vector_store"
            self.vector_store.save(str(store_file))

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
            query_vector = await self.embedding_model.encode_single(query)

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
            if self.vector_store:
                # 保存当前状态
                store_file = self.vector_store_path / "vector_store"
                self.vector_store.save(str(store_file))

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
