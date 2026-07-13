from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import LLM_API_KEY
from app.models.document import DocumentRecord
from app.models.schemas import ChatRequest
from app.services.retriever import retriever
from app.services.llm_handler import llm_handler
from app.services.embedding_service import embedding_service

router = APIRouter(prefix="/api", tags=["问答"])


@router.post("/chat")
def chat(req: ChatRequest):
    """向知识库提问"""
    sources = retriever.retrieve(req.question)
    result = llm_handler.generate(req.question, sources)

    source_items = [{
        "chunk_id": s.get("chunk_id", ""),
        "content": s.get("content", "")[:200] + ("..." if len(s.get("content", "")) > 200 else ""),
        "score": s.get("score", 0),
    } for s in sources]

    return {
        "question": req.question,
        "answer": result["answer"],
        "sources": source_items[:3],
        "source_count": len(sources),
        "model_used": result.get("model_used", ""),
        "latency_ms": result.get("latency_ms", 0),
    }


@router.get("/chat/history")
def chat_history(db: Session = Depends(get_db)):
    """对话历史"""
    from app.models.document import QueryLog
    logs = db.query(QueryLog).order_by(QueryLog.created_at.desc()).limit(50).all()
    return [
        {"id": l.id, "question": l.question, "answer": (l.answer or "")[:300],
         "model_used": l.model_used, "created_at": l.created_at.isoformat()}
        for l in logs
    ]


@router.get("/stats")
def stats(db: Session = Depends(get_db)):
    """统计信息"""
    doc_count = db.query(DocumentRecord).count()
    chunk_count = embedding_service.count()
    mode = "AI智能模式" if (LLM_API_KEY and LLM_API_KEY != "your-api-key-here") else "本地搜索模式"

    return {"doc_count": doc_count, "chunk_count": chunk_count, "mode": mode}
