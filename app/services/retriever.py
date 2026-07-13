from typing import List
from app.services.embedding_service import embedding_service
from app.core.config import TOP_K


class Retriever:
    """
    检索引擎 - 搜索最相关的知识片段

    采用混合策略:
    1. 语义搜索 (向量相似度) - 理解意思
    2. 关键词匹配 (BM25) - 精确命中
    结果融合后返回
    """

    def __init__(self):
        self._all_chunks = []       # 所有块的文字
        self._all_doc_ids = []      # 每个块对应的文档ID

    def _rebuild_keyword_index(self):
        """从 ChromaDB 加载所有块，构建关键词索引"""
        try:
            collection = embedding_service._get_collection()
            all_data = collection.get(include=["documents", "metadatas"])
            self._all_chunks = all_data.get("documents", [])
            self._all_doc_ids = [
                m.get("doc_id", "") for m in all_data.get("metadatas", [])
            ]
        except Exception:
            self._all_chunks = []
            self._all_doc_ids = []

    def retrieve(self, query: str, top_k: int = None) -> List[dict]:
        """
        混合检索 - 语义 + 关键词
        """
        top_k = top_k or TOP_K

        # 1. 语义搜索 (理解意思的搜索)
        vec_results = embedding_service.search(query, top_k=top_k * 2)

        # 2. 关键词匹配
        kw_results = self._keyword_search(query, top_k=top_k)

        # 3. 融合去重
        merged = self._merge(vec_results, kw_results, top_k)

        return merged

    def _keyword_search(self, query: str, top_k: int) -> List[dict]:
        """简单的关键词匹配 - 看哪些块包含问的字眼"""
        self._rebuild_keyword_index()

        if not self._all_chunks:
            return []

        # 对每个块打分
        scored = []
        query_chars = set(query)

        for i, chunk_text in enumerate(self._all_chunks):
            score = 0
            # 包含越多的关键字，分数越高
            for char in query_chars:
                if char in chunk_text:
                    score += 1
            # 关键词连续匹配加分
            for word in query.split():
                if word in chunk_text:
                    score += 3
            if score > 0:
                scored.append({
                    "chunk_id": f"kw_{i}",
                    "content": chunk_text[:500],
                    "doc_id": self._all_doc_ids[i] if i < len(self._all_doc_ids) else "",
                    "chunk_index": i,
                    "score": min(score / 20, 0.5),
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _merge(self, vec_results: List[dict], kw_results: List[dict], top_k: int) -> List[dict]:
        """融合并去重两个搜索结果"""
        seen_contents = set()
        merged = []

        # 语义搜索结果先加入（更可靠）
        for r in vec_results:
            key = r["content"][:100]
            if key not in seen_contents:
                seen_contents.add(key)
                merged.append(r)

        # 关键词结果补充
        for r in kw_results:
            key = r["content"][:100]
            if key not in seen_contents:
                seen_contents.add(key)
                r["score"] = r.get("score", 0) * 0.8  # 关键词结果权重降低
                merged.append(r)

        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:top_k]


retriever = Retriever()
