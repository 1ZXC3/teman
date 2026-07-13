from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles  # type: ignore
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse  # type: ignore

from app.core.config import DATA_DIR
from app.core.database import engine, Base
from app.models.document import DocumentRecord  # noqa: F401 - 注册模型
from app.api import documents, chat

# 创建静态文件目录
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时自动创建数据库表并预加载AI模型"""
    Base.metadata.create_all(bind=engine)
    print("=" * 50)
    print("  我的知识助手 启动中...")
    print("=" * 50)

    # 预加载模型（首次需下载 ~100MB，之后秒启动）
    from app.services.embedding_service import embedding_service
    embedding_service.preload()

    print("-" * 50)
    print("  浏览器访问: http://localhost:8000")
    print("  DeepSeek AI 已就绪")
    print("=" * 50)
    yield
    engine.dispose()


app = FastAPI(
    title="📚 我的知识助手",
    version="2.0",
    description="一个简单好用的个人知识库问答系统",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# API 路由
app.include_router(documents.router)
app.include_router(chat.router)

# 前端页面
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    """主页 - 知识问答界面"""
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
