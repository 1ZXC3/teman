import os
import threading
from typing import List
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from app.core.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, EMBEDDING_DEVICE

# 国内加速：使用 HF 镜像站
if not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class EmbeddingService:
    """向量服务 - 把文字变成数字指纹，方便搜索"""

    def __init__(self):
        self._model = None
        self._client = None
        self._lock = threading.Lock()

    def preload(self):
        """启动时预加载模型，避免首次请求卡住"""
        print("正在下载/加载AI模型... (首次约1-2分钟)")
        self._get_model()
        print("模型就绪!")

    def _get_model(self):
        if self._model is None:
            self._model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
        return self._model

    def encode(self, texts, **kwargs):
        with self._lock:
            return self._get_model().encode(texts, **kwargs)

    def _get_collection(self):
        """每次新建客户端连接"""
        client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_document(self, doc_id: str, chunks: List[str]):
        """把文档的每个块加入向量库"""
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
        """搜索最相关的文字片段"""
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
        """删除文档的所有块"""
        with self._lock:
            collection = self._get_collection()
            existing = collection.get(where={"doc_id": doc_id})
            if existing["ids"]:
                collection.delete(ids=existing["ids"])

    def count(self) -> int:
        collection = self._get_collection()
        return collection.count()


embedding_service = EmbeddingService()
