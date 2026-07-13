import os
from pathlib import Path

# ========== 项目路径 ==========
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # config.py 在 app/core/ 下，往上3层到项目根
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"

# 自动创建需要的文件夹
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# ========== 数据库: MySQL ==========
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "rag_user")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "rag_password_2024")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rag_knowledge_base")

DATABASE_URL = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
    "?charset=utf8mb4"
)

# ========== ChromaDB (向量存储) ==========
CHROMA_DIR = str(DATA_DIR / "chroma_db")
COLLECTION_NAME = "knowledge_chunks"

# ========== Embedding 模型 ==========
# 已从 ModelScope 下载到本地，无需联网
EMBEDDING_MODEL = str(DATA_DIR / "bge-small-zh" / "models" / "AI-ModelScope--bge-small-zh-v1.5" / "snapshots" / "master")
EMBEDDING_DEVICE = "cpu"

# ========== LLM 设置 (DeepSeek) ==========
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# ========== 文档处理 ==========
CHUNK_SIZE = 500          # 每块最多500字
CHUNK_OVERLAP = 50        # 块之间保留50字重叠
TOP_K = 5                 # 每次检索返回5个最相关的片段

# ========== 文件上传 ==========
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
