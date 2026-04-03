"""
LangChain 跨版本兼容导入。

- LangChain 0.3+：`Document` 在 `langchain_core.documents`，`ChatPromptTemplate` 在 `langchain_core.prompts`，
  `RecursiveCharacterTextSplitter` 在 `langchain_text_splitters`。
- 旧版：部分符号在 `langchain.schema` / `langchain.prompts` / `langchain.text_splitter`。

新机器迁移时请务必：`pip install -r requirements.txt`（含 langchain-core、langchain-text-splitters）。
"""

from __future__ import annotations

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document  # LangChain < 0.3
    except ImportError as e:
        raise ImportError(
            "无法导入 LangChain Document。请安装: pip install -U \"langchain-core>=0.2\" "
            "或完整执行 pip install -r requirements.txt"
        ) from e

try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    try:
        from langchain.prompts import ChatPromptTemplate
    except ImportError as e:
        raise ImportError(
            "无法导入 ChatPromptTemplate。请安装: pip install -U \"langchain-core>=0.2\""
        ) from e

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitters import RecursiveCharacterTextSplitter
        except ImportError as e:
            raise ImportError(
                "无法导入 RecursiveCharacterTextSplitter。请安装: "
                'pip install -U "langchain-text-splitters>=0.2"'
            ) from e

__all__ = ["Document", "ChatPromptTemplate", "RecursiveCharacterTextSplitter"]
