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
    """启动时自动创建数据库表"""
    Base.metadata.create_all(bind=engine)
    print("=" * 50)
    print("📚 我的知识助手 启动成功！")
    print("=" * 50)
    print(f"🌐 打开浏览器访问: http://localhost:8000")
    print(f"📁 数据存储位置: {DATA_DIR}")
    print(f"📄 上传文件目录: {Path.cwd() / 'uploads'}")
    print("-" * 50)
    print("💡 提示: 没有 API Key 也能用本地搜索模式！")
    print("   如果要用 AI 总结功能，请设置环境变量 LLM_API_KEY")
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
