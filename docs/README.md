# 注册文档审核工具 (AI Check Word)

基于 RAG（检索增强生成）技术的注册文档智能审核工具。通过训练法规、标准和项目文件构建知识库，自动审核注册文档并输出结构化审核报告。

## 核心功能

- **知识库训练** — 支持 PDF / Word / Excel / TXT / Markdown 格式，将法规和标准文件向量化存储
- **智能审核** — 基于知识库对待审核文档进行多维度审核（合规性、完整性、一致性、准确性、格式规范）
- **批量处理** — 支持同时上传和审核多个文件
- **结构化输出** — 每个审核点包含类别、严重程度、位置、描述、法规依据和修改建议
- **Agent API** — 提供 REST API 接口，可被其他项目集成调用
- **多知识库** — 支持创建多个独立知识库，适用于不同项目

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录下：

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API Key 或 MySQL 等配置
```

`.env` 关键配置见根目录 `.env.example`（OpenAI / Ollama / MySQL 等）。

### 3. 启动 Web UI

在项目根目录执行：

```bash
streamlit run src/app.py
```

浏览器访问 `http://localhost:8501`

### 4. 启动 API 服务（供其他项目调用）

在项目根目录执行：

```bash
python -m src.api.server
```

或：

```bash
python -m src.api
```

API 文档访问 `http://localhost:8000/docs`

## 使用流程

```
1. 训练阶段：上传法规、标准、程序文件 → 构建知识库
2. 审核阶段：上传待审核文档 → AI 自动审核
3. 获取结果：查看/下载结构化审核报告（JSON / HTML / PDF / Word / Markdown）
```

## API 接口说明

### 训练接口

```bash
# 上传文件训练
curl -X POST http://localhost:8000/train/upload \
  -F "files=@法规文件.pdf" \
  -F "collection=regulations"

# 从目录训练
curl -X POST http://localhost:8000/train/directory \
  -F "dir_path=./training_docs" \
  -F "collection=regulations"
```

### 审核接口

```bash
# 上传文件审核
curl -X POST http://localhost:8000/review/upload \
  -F "files=@待审核文档.docx" \
  -F "collection=regulations"

# 文本审核
curl -X POST http://localhost:8000/review/text \
  -H "Content-Type: application/json" \
  -d '{"text": "待审核文本内容", "collection": "regulations"}'
```

### 知识库接口

```bash
# 查询知识库
curl -X POST http://localhost:8000/knowledge/search \
  -H "Content-Type: application/json" \
  -d '{"query": "产品注册需要哪些资料", "top_k": 5}'

# 查看状态
curl http://localhost:8000/status?collection=regulations
```

## 作为 Agent 在代码中调用

在项目根目录或已将项目根加入 `PYTHONPATH` 时：

```python
from src.core.agent import ReviewAgent

# 初始化 Agent
agent = ReviewAgent(collection_name="my_project")

# 训练
agent.train("./training_docs/法规.pdf")
agent.train("./training_docs/标准/")  # 支持目录

# 审核
report = agent.review("./待审核文档.docx")
print(report)

# 批量审核
reports = agent.review_batch(["doc1.pdf", "doc2.docx"])

# 查询知识库
results = agent.search_knowledge("注册资料要求")
```

## 项目结构

```
aicheckword/
├── config/                 # 配置
│   ├── __init__.py
│   └── settings.py         # 应用与 AI 配置
├── docs/                   # 文档
│   └── README.md           # 本说明
├── src/                    # 业务代码
│   ├── app.py              # Streamlit Web UI 入口
│   ├── core/               # 核心逻辑
│   │   ├── agent.py
│   │   ├── db.py           # MySQL 与操作记录
│   │   ├── document_loader.py
│   │   ├── knowledge_base.py
│   │   ├── reviewer.py
│   │   ├── cursor_agent.py
│   │   └── report_export.py
│   └── api/                # REST API
│       ├── server.py
│       └── __main__.py
├── requirements.txt
├── .env.example             # 环境变量模板（根目录）
├── knowledge_store/        # 向量库持久化
├── training_docs/          # 训练文档目录
└── uploads/                # 临时上传目录
```

## 技术架构

- **文档处理**：LangChain + PyPDF / python-docx / openpyxl
- **向量存储**：ChromaDB（本地持久化）
- **持久化**：MySQL（配置、操作记录、审核报告、知识库文档记录）
- **大语言模型**：OpenAI / Ollama / Cursor Cloud Agents
- **Web UI**：Streamlit
- **API 服务**：FastAPI

用户上传法规/标准 → 文档加载 → 分块 → 向量化 → ChromaDB + MySQL  
用户上传待审核文档 → 文档加载 → 检索相关法规 → LLM 审核 → 结构化报告
