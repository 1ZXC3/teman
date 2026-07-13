# 企业知识库 RAG 问答系统

基于检索增强生成（RAG）架构的企业知识库问答系统。

## 系统架构

- **接入层**: FastAPI RESTful 接口
- **文档处理层**: 多格式解析 → 智能分块 → 向量化存储
- **检索层**: Milvus 向量检索 + BM25 关键词融合
- **生成层**: 双模型 LLM 路由 + 自动 Fallback
- **统计层**: Token 用量记录、查询指标上报

## 快速启动

### 1. 克隆项目

```bash
cd rag-knowledge-base
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key
```

### 3. Docker Compose 启动

```bash
docker-compose up -d
```

### 4. 访问 API 文档

```
http://localhost:8000/docs
```

## 项目结构

```
rag-knowledge-base/
├── app/
│   ├── api/                 # API 路由
│   │   ├── documents.py     # 文档管理接口
│   │   ├── chat.py          # 问答接口
│   │   └── stats.py         # 统计接口
│   ├── core/                # 核心配置
│   │   ├── config.py        # 应用配置
│   │   ├── database.py      # 数据库连接
│   │   └── security.py      # 安全与鉴权
│   ├── models/              # 数据模型
│   │   ├── document.py      # ORM 模型
│   │   └── schemas.py       # Pydantic 模型
│   ├── services/            # 业务服务
│   │   ├── document_processor.py  # 文档解析
│   │   ├── embedding_service.py   # 嵌入向量
│   │   ├── retriever.py           # 混合检索
│   │   ├── llm_handler.py         # LLM 处理
│   │   └── stats_service.py       # 统计服务
│   ├── utils/               # 工具函数
│   │   └── token_counter.py       # Token 计数
│   └── main.py              # 应用入口
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```
