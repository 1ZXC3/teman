from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.document import DocumentRecord
from app.services.document_processor import process_document
from app.services.embedding_service import embedding_service

router = APIRouter(prefix="/api/documents", tags=["文档"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档"""
    if not file.filename:
        raise HTTPException(400, "文件名不能为空")

    # 检查格式
    from app.core.config import ALLOWED_EXTENSIONS
    from pathlib import Path
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"不支持的格式: {ext}，支持: {', '.join(ALLOWED_EXTENSIONS)}")

    content = await file.read()

    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(400, "文件太大，最大 20MB")

    # 处理文档
    text, chunks = process_document(content, file.filename)

    if not chunks:
        raise HTTPException(400, "文档中未提取到文字内容")

    # 保存到数据库
    db = next(get_db())
    try:
        doc = DocumentRecord(
            filename=file.filename,
            file_type=ext,
            chunk_count=len(chunks),
        )
        db.add(doc)
        db.flush()

        # 存入向量库
        embedding_service.add_document(doc.id, chunks)

        db.commit()
        return {"id": doc.id, "filename": doc.filename, "chunk_count": len(chunks)}
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"处理失败: {e}")
    finally:
        db.close()


@router.get("")
def list_documents():
    """获取文档列表"""
    db = next(get_db())
    try:
        docs = db.query(DocumentRecord).order_by(DocumentRecord.upload_time.desc()).all()
        return [
            {"id": d.id, "filename": d.filename, "file_type": d.file_type,
             "chunk_count": d.chunk_count, "upload_time": d.upload_time.isoformat()}
            for d in docs
        ]
    finally:
        db.close()


@router.delete("/{doc_id}")
def delete_document(doc_id: str):
    """删除文档"""
    db = next(get_db())
    try:
        doc = db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).first()
        if not doc:
            raise HTTPException(404, "文档不存在")

        # 从向量库删除
        embedding_service.delete_document(doc_id)

        # 从数据库删除
        db.delete(doc)
        db.commit()
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"删除失败: {e}")
    finally:
        db.close()


@router.delete("/clear")
def clear_all_documents():
    """清空所有文档"""
    db = next(get_db())
    try:
        # 逐个删除向量数据
        docs = db.query(DocumentRecord).all()
        for doc in docs:
            embedding_service.delete_document(doc.id)
            db.delete(doc)
        db.commit()
        return {"ok": True, "count": len(docs)}
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"清空失败: {e}")
    finally:
        db.close()
