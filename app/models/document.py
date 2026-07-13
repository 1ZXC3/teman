import uuid
import datetime
from sqlalchemy import Column, String, Integer, DateTime, Text
from app.core.database import Base


def gen_uid():
    return uuid.uuid4().hex[:12]


class DocumentRecord(Base):
    """文档记录表"""
    __tablename__ = "documents"

    id = Column(String(12), primary_key=True, default=gen_uid)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(20), nullable=False)
    chunk_count = Column(Integer, default=0)
    upload_time = Column(DateTime, default=datetime.datetime.now)
