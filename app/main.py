import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.core.config import DATA_DIR
from app.core.logger import setup_logging, get_logger
from app.core.database import engine, Base
from app.models.document import DocumentRecord  # noqa: F401
from app.api import documents, chat

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

setup_logging()
logger = get_logger(__name__)

ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化数据库和AI模型"""
    Base.metadata.create_all(bind=engine)
    logger.info("数据库表已就绪")

    from app.services.embedding_service import embedding_service
    embedding_service.preload()

    logger.info(f"服务启动成功 -> http://localhost:8000")
    yield
    engine.dispose()


app = FastAPI(
    title="我的知识助手",
    version="2.1",
    description="个人知识库RAG问答系统",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
