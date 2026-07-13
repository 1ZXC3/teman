# 我的知识助手 - RAG 知识库问答系统

基于检索增强生成（RAG）架构的个人知识库问答系统。

## 系统架构

- **接入层**: FastAPI RESTful 接口 + 浏览器网页界面
- **文档处理层**: PDF/Word/TXT 解析 → 智能分块
- **检索层**: ChromaDB 向量检索 + 关键词匹配
- **生成层**: DeepSeek API 智能回答
- **存储层**: MySQL (文档元数据) + ChromaDB (向量数据)

## 快速启动

### 1. 配置 API Key

```bash
# Windows PowerShell:
set LLM_API_KEY=sk-你的deepseek密钥
```

或在项目根目录创建 `.env` 文件：
```
LLM_API_KEY=sk-你的deepseek密钥
```

不设置 API Key 也能用本地搜索模式。

### 2. 启动

```bash
cd rag-knowledge-base
pip install -r requirements.txt
python start.py
```

浏览器打开 http://localhost:8000

### 3. Docker 启动 (可选)

```bash
docker-compose up -d
```

## 项目结构

```
rag-knowledge-base/
├── app/
│   ├── api/                     # API 路由
│   │   ├── documents.py         # 文档上传/删除/列表
│   │   └── chat.py              # 问答 + 统计
│   ├── core/                    # 核心配置
│   │   ├── config.py            # 应用配置
│   │   └── database.py          # 数据库连接
│   ├── models/                  # 数据模型
│   │   ├── document.py          # ORM 模型
│   │   └── schemas.py           # Pydantic 模型
│   ├── services/                # 业务服务
│   │   ├── document_processor.py  # 文档解析 + 分块
│   │   ├── embedding_service.py   # 向量化 + ChromaDB
│   │   ├── retriever.py          # 混合检索
│   │   └── llm_handler.py        # DeepSeek 调用
│   ├── static/
│   │   └── index.html           # 网页前端
│   └── main.py                  # FastAPI 入口
├── docker-compose.yml           # Docker 部署 (app + MySQL)
├── Dockerfile
├── requirements.txt
├── start.py                     # 一键启动脚本
└── README.md
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/documents/upload | 上传文档 |
| GET | /api/documents | 文档列表 |
| DELETE | /api/documents/{id} | 删除文档 |
| POST | /api/chat | 向知识库提问 |
| GET | /api/chat/history | 对话历史 |
| GET | /api/stats | 系统统计 |

## 技术栈

Python 3.11+ | FastAPI | ChromaDB | MySQL | Sentence-Transformers | DeepSeek API
