# 训练与分块逻辑说明

## 一、训练的整体实现逻辑

**训练不只是分块**，完整流程是：**加载文档 → 分块 → 向量化（Embedding）→ 写入向量库 + 落库 MySQL**。

### 1. 流程概览

```
用户上传/选择文件
    ↓
load_single_file / load_directory（按格式解析为 Document 列表）
    ↓
split_documents（按 chunk_size / chunk_overlap 分块）
    ↓
add_documents / add_documents_with_progress
    ├── 对每个块调用 Embedding 模型得到向量
    ├── 写入 ChromaDB 向量库（便于后续相似度检索）
    └── 将块内容 + 元数据写入 MySQL（knowledge_docs / checkpoint_docs / project_knowledge_docs）
```

### 2. 各步骤职责

| 步骤 | 实现位置 | 作用 |
|------|----------|------|
| 加载 | `document_loader.load_single_file` / `load_directory` | 按后缀选 Loader（PDF/Word/Excel/TXT/MD），解析成 LangChain `Document` 列表。**PDF 仅支持文本型**（见下） |

### 3. PDF 说明（含图片/扫描版）

- **当前实现**：使用 `PyPDFLoader`（基于 pypdf），只提取 PDF 中**嵌入的文字**，不识别图片里的文字（无 OCR）。
- **扫描件/纯图片 PDF**：无法提取到文字，会报错或表现为“卡住”。加载时会检测：若整份 PDF 提取文字少于 20 字，会直接报错并提示“可能为扫描件/图片版，建议先 OCR 或使用文本型 PDF”。
- **大文件**：单份 PDF 最多处理前 **500 页**，避免页数过多导致长时间无响应。

### 4. Word 说明（.docx 与旧版 .doc）

- **.docx**：优先使用 `Docx2txtLoader`（轻量、无需额外依赖）。若文件实为旧版 .doc 被改名为 .docx 或损坏（会报 `BadZipFile`），则自动改用 `UnstructuredWordDocumentLoader` 解析。
- **旧版 .doc（Word 97-2003）**：先尝试 `UnstructuredWordDocumentLoader`；若失败或解析结果为空，则自动用系统已安装的 **LibreOffice** 将 .doc 转为 .docx 后再解析。请确保 LibreOffice 已安装（Windows 常见路径：`C:\Program Files\LibreOffice\program\soffice.exe`）；若安装在其他目录，可设置环境变量 **`LIBREOFFICE_PATH`** 指向 `soffice.exe` 或其所在目录。

| 分块 | `document_loader.split_documents` | 用 `RecursiveCharacterTextSplitter` 按长度与分隔符切分，得到多个“块” |
| 向量化 | `knowledge_base.add_documents` 内部 | 调用 Ollama/OpenAI 的 Embedding 接口，得到每个块的向量 |
| 存储 | `KnowledgeBase.vectorstore` + `db.save_*_docs` | 向量写入 Chroma，文本+元数据写入 MySQL，便于持久化与统计 |

所以：**训练 = 分块 + 向量化 + 双写存储**，分块只是其中一步。

---

## 二、分块的实现逻辑

### 1. 使用的组件

- **类**：`RecursiveCharacterTextSplitter`（LangChain）
- **配置**：`config/settings.py` 中的 `chunk_size`、`chunk_overlap`（默认 1000、200）

### 2. 核心代码（document_loader.py）

```python
def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """将文档分块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,   # 每块目标长度，如 1000 字符
        chunk_overlap=chunk_overlap or settings.chunk_overlap,  # 块与块之间重叠长度，如 200
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],  # 优先按这些分隔符切，尽量不截断句
    )
    return splitter.split_documents(documents)
```

### 3. 分块策略简述

- **chunk_size**：单块的目标长度（字符数），超过则继续切。
- **chunk_overlap**：相邻两块之间的重叠长度，避免句首/句尾被硬切断，有利于上下文连贯。
- **separators**：按顺序尝试的分隔符；先尽量用 `\n\n` 分段，再 `\n`，再 `。`、`；`、`，`、空格，最后才用空字符串（按字符切）。这样尽量在“自然边界”处切块，减少断句。

因此：**分块 = 按长度限制 + 重叠 + 中文/英文分隔符优先的递归切分**。

---

## 三、Cursor Agent 在系统里的角色

### 1. 定位

Cursor Agent 是**可选的“大模型”提供方**之一，与 **Ollama（本地）**、**OpenAI（API）** 并列；用户侧栏选择「Cursor Agent (Cloud API)」时启用。

### 2. 用在哪里（只做“推理”，不做向量）

- **不参与训练/向量化**  
  训练时的 Embedding 仍由 **Ollama 或 OpenAI** 完成（Cursor 模式下由「向量化使用」选择 ollama/openai）。
- **只参与“需要生成文本”的环节**：
  - **文档审核**（第三步）：根据审核点知识 + 待审文档，生成审核意见（`reviewer.py` 里 `review_text` / `review_file`）。
  - **审核报告摘要**：根据审核点列表生成一段总结（`_generate_summary`）。

也就是说：**Cursor 在系统里扮演“审核与摘要的 LLM”**，不参与分块、不参与向量化，只负责“根据上下文生成审核结果和摘要”。

### 3. 调用方式

- **Ollama / OpenAI**：通过 LangChain 的 `ChatOllama` / `ChatOpenAI`，直接 `prompt | llm` 调用。
- **Cursor**：不走标准 Chat 接口，而是通过 **Cursor Cloud Agents API**（`cursor_agent.py`）：
  1. `launch_agent(prompt_text)`：把完整任务描述（审核/摘要）当作一条 prompt 发给 Cursor，创建 Agent 任务。
  2. `poll_until_finished(agent_id)`：轮询任务状态直到完成。
  3. `get_last_assistant_reply(agent_id)`：从对话里取出助手的最后一条回复，当作“模型输出”。
  4. 对这份回复再做解析（如 `_parse_audit_points`）得到结构化审核点或摘要。

所以：**Cursor Agent 在系统里 = 文档审核与摘要的“云端大模型”提供方**，与 Ollama/OpenAI 二选一（或选 Cursor），训练与分块逻辑与是否使用 Cursor 无关。

---

## 四、小结

| 问题 | 简要回答 |
|------|----------|
| 训练是不是只做分块？ | 否。训练 = 加载 → **分块** → **向量化** → 写入 Chroma + MySQL。 |
| 分块逻辑是什么？ | 使用 `RecursiveCharacterTextSplitter`，按 `chunk_size`/`chunk_overlap` 和分隔符 `["\n\n","\n","。","；","，"," ",""]` 递归切分，尽量在句/段边界切。 |
| Cursor Agent 扮演什么角色？ | 仅作为“审核 + 摘要”的 LLM；不参与分块、不参与向量化；与 Ollama/OpenAI 二选一用于文档审核和报告摘要。 |
