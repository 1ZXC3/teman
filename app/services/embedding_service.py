from typing import List
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from app.core.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, EMBEDDING_DEVICE


class EmbeddingService:
    """
    向量服务 - 把文字变成"数字指纹"，方便搜索
    使用 ChromaDB (一个轻量内置数据库，不需要额外安装软件)
    """

    def __init__(self):
        self._model = None
        self._client = None
        self._collection = None

    # ========== 模型 (首次使用自动下载) ==========
    @property
    def model(self):
        if self._model is None:
            print("🔄 正在加载 AI 模型... (首次使用需要下载，请耐心等待)")
            self._model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
            print("✅ 模型加载完成！")
        return self._model

    # ========== ChromaDB 连接 ==========
    @property
    def client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=CHROMA_DIR,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ========== 核心功能 ==========
    def add_document(self, doc_id: str, chunks: List[str]):
        """把文档的每个块加入向量库"""
        if not chunks:
            return

        # 生成向量
        embeddings = self.model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)

        # 存入 ChromaDB
        self.collection.add(
            ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=[
                {"doc_id": doc_id, "chunk_index": i, "chunk_count": len(chunks)}
                for i in range(len(chunks))
            ],
        )
        print(f"✅ 已存储 {len(chunks)} 个文字片段")

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """搜索最相关的文字片段"""
        query_vec = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)

        results = self.collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        search_results = []
        for i in range(len(results["ids"][0])):
            search_results.append({
                "chunk_id": results["ids"][0][i],
                "content": results["documents"][0][i] if results["documents"] else "",
                "doc_id": results["metadatas"][0][i].get("doc_id", "") if results["metadatas"] else "",
                "chunk_index": results["metadatas"][0][i].get("chunk_index", i) if results["metadatas"] else i,
                "score": round(1 - results["distances"][0][i], 4),  # cosine距离 → 相似度
            })

        return search_results

    def delete_document(self, doc_id: str):
        """删除文档的所有块"""
        # 查找该文档的所有块ID
        existing = self.collection.get(where={"doc_id": doc_id})
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])

    def count(self) -> int:
        """统计存储了多少片段"""
        return self.collection.count()


# 全局单例
embedding_service = EmbeddingService()
