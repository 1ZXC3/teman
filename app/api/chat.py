from fastapi import APIRouter
from app.models.schemas import ChatRequest
from app.services.retriever import retriever
from app.services.llm_handler import llm_handler
from app.core.config import LLM_API_KEY

router = APIRouter(prefix="/api", tags=["问答"])


@router.post("/chat")
def chat(req: ChatRequest):
    """向知识库提问"""
    # 1. 搜索相关片段
    sources = retriever.retrieve(req.question)

    # 2. AI 生成回答
    result = llm_handler.generate(req.question, sources)

    # 3. 构建来源信息
    source_items = []
    for s in sources:
        source_items.append({
            "chunk_id": s.get("chunk_id", ""),
            "content": s.get("content", "")[:200] + "..." if len(s.get("content", "")) > 200 else s.get("content", ""),
            "score": s.get("score", 0),
        })

    return {
        "question": req.question,
        "answer": result["answer"],
        "sources": source_items[:3],
        "source_count": len(sources),
        "model_used": result.get("model_used", ""),
        "latency_ms": result.get("latency_ms", 0),
    }


@router.get("/stats")
def stats():
    """获取统计信息"""
    from app.services.embedding_service import embedding_service
    from app.models.document import DocumentRecord
    from app.core.database import SessionLocal

    db = SessionLocal()
    try:
        doc_count = db.query(DocumentRecord).count()
    finally:
        db.close()

    chunk_count = embedding_service.count()

    mode = "本地搜索模式"
    if LLM_API_KEY and LLM_API_KEY != "your-api-key-here":
        mode = "AI 智能模式 ✨"

    return {
        "doc_count": doc_count,
        "chunk_count": chunk_count,
        "mode": mode,
    }
