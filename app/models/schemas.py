from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class DocInfo(BaseModel):
    id: str
    filename: str
    file_type: str
    chunk_count: int
    upload_time: datetime


class SourceItem(BaseModel):
    chunk_id: str
    content: str
    score: float


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list = []
    source_count: int = 0
    model_used: str = ""
    latency_ms: float = 0
