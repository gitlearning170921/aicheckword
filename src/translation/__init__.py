"""
文档翻译子模块：面向 FDA 医疗器械认证，仅中文→英文逐句翻译，保持原格式与结构。
复用 aicheckword 的 config.settings 与 llm_factory，可选复用知识库（词条/法规/案例）做参考。
"""

from .parser import parse_document, SUPPORTED_EXTENSIONS
from .segment import segment_chinese_sentences
from .translator import translate_sentences
from .pipeline import translate_file, translate_path

__all__ = [
    "parse_document",
    "segment_chinese_sentences",
    "translate_sentences",
    "translate_file",
    "translate_path",
    "SUPPORTED_EXTENSIONS",
]
