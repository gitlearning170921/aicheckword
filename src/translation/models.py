"""翻译模块用到的数据结构。"""
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional

# 首期支持格式
SUPPORTED_EXTENSIONS = (".docx", ".txt", ".xlsx")


@dataclass
class TextBlock:
    """文档中的一个可替换文本块（段落、表格单元格、标题或整行）。"""
    block_type: str  # "paragraph" | "table_cell" | "heading" | "line"
    path: Tuple[Any, ...]  # 定位：如 (para_index,) 或 (table_idx, row, col)
    original_text: str
    translated_text: str = ""  # 回填时写入


@dataclass
class SegmentResult:
    """分句结果：一段原文被拆成多个片段，标记是否需翻译。"""
    text: str
    need_translate: bool  # 含中文/外文则需翻译（由目标语言决定）
    # 混合中英/德时按语言拆成的 (run_text, is_cjk)；仅当 need_translate 且同时含 CJK 与拉丁时设置
    runs: Optional[List[Tuple[str, bool]]] = None  # (run_text, is_cjk)
