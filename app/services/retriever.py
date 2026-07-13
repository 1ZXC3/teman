from typing import List
from app.services.embedding_service import embedding_service
from app.core.config import TOP_K


class Retriever:
    """检索引擎 - 纯向量语义搜索"""

    def retrieve(self, query: str, top_k: int = None) -> List[dict]:
        """语义搜索，返回最相关的知识片段"""
        top_k = top_k or TOP_K
        return embedding_service.search(query, top_k=top_k)


retriever = Retriever()
