import threading
from typing import List
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from app.core.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, EMBEDDING_DEVICE


class EmbeddingService:
    """向量服务 - 使用本地模型"""

    def __init__(self):
        self._model = None
        self._client = None
        self._collection = None
        self._lock = threading.RLock()

    def preload(self):
        """启动时初始化模型和 ChromaDB 连接"""
        print("正在加载AI模型...")
        self._get_model()
        print("正在初始化向量库...")
        self._get_collection()
        print("就绪!")

    def _get_model(self):
        if self._model is None:
            try:
                self._model = SentenceTransformer(
                    EMBEDDING_MODEL, device=EMBEDDING_DEVICE, local_files_only=True
                )
            except Exception:
                print("本地模型未找到，从网络下载...")
                self._model = SentenceTransformer(
                    "BAAI/bge-small-zh-v1.5", device=EMBEDDING_DEVICE
                )
        return self._model

    def _get_collection(self):
        """单例 ChromaDB 客户端和集合"""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=CHROMA_DIR,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def encode(self, texts, **kwargs):
        with self._lock:
            return self._get_model().encode(texts, **kwargs)

    def add_document(self, doc_id: str, chunks: List[str]):
        if not chunks:
            return
        with self._lock:
            embeddings = self.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
            collection = self._get_collection()
            collection.add(
                ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
                documents=chunks,
                embeddings=embeddings.tolist(),
                metadatas=[{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))],
            )

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        with self._lock:
            query_vec = self.encode([query], normalize_embeddings=True, show_progress_bar=False)
            collection = self._get_collection()
            results = collection.query(
                query_embeddings=query_vec.tolist(),
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

        if not results["ids"] or not results["ids"][0]:
            return []

        search_results = []
        for i in range(len(results["ids"][0])):
            dist = results["distances"][0][i]
            search_results.append({
                "chunk_id": results["ids"][0][i],
                "content": results["documents"][0][i] if results["documents"] else "",
                "doc_id": results["metadatas"][0][i].get("doc_id", "") if results["metadatas"] else "",
                "chunk_index": i,
                "score": round(1 - dist, 4),
            })
        return search_results

    def delete_document(self, doc_id: str):
        with self._lock:
            collection = self._get_collection()
            existing = collection.get(where={"doc_id": doc_id})
            if existing["ids"]:
                collection.delete(ids=existing["ids"])

    def count(self) -> int:
        return self._get_collection().count()


embedding_service = EmbeddingService()
