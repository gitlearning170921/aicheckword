"""
Cursor skills / rules 增量更新器。

目的：
- 根据用户输入的“文件块补丁”，把内容合并到本地仓库的：
  - .cursor/skills/**/SKILL.md
  - .cursor/rules/*.mdc
- 对合并采取去重策略：按“段落”维度去重（连续空行分隔）。
- 每次写入都保留原有内容；默认使用 append 合并模式，除非用户显式声明 @replace。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class PatchUpdate:
    rel_path: str
    mode: str  # "append" | "replace"
    content: str


_HEADER_RE = re.compile(r"^\s*###\s*FILE\s*:\s*(?P<path>\S+)\s*(?P<opts>.*)\s*$")


def _split_paragraphs(text: str) -> List[str]:
    """
    段落切分：用 2 个以上换行隔开。
    保留原段落内容（trim 后），用于“精确段落去重”。
    """
    t = (text or "").strip()
    if not t:
        return []
    paras = re.split(r"\n{2,}", t)
    out: List[str] = []
    for p in paras:
        p2 = (p or "").strip()
        if p2:
            out.append(p2)
    return out


def parse_patch_updates(raw: str) -> List[PatchUpdate]:
    """
    解析用户输入的补丁文本。

    约定格式（支持多文件块）：
    - 每个文件块以一行 header 开头：
      ### FILE: .cursor/skills/foo/SKILL.md
      ### FILE: .cursor/rules/document-authoring-and-audit.mdc @replace
    - header 之后到下一个 header 之前为内容。
    """
    text = raw or ""
    if not text.strip():
        return []

    lines = text.splitlines()
    blocks: List[tuple[str, str, str]] = []  # (path, mode, content)

    cur_path: Optional[str] = None
    cur_mode: str = "append"
    cur_content: List[str] = []

    def _flush():
        nonlocal cur_path, cur_mode, cur_content
        if not cur_path:
            return
        content = "\n".join(cur_content).strip("\n")
        if content.strip():
            blocks.append((cur_path, cur_mode, content))
        cur_path = None
        cur_mode = "append"
        cur_content = []

    for line in lines:
        m = _HEADER_RE.match(line)
        if m:
            _flush()
            cur_path = (m.group("path") or "").strip()
            opts = (m.group("opts") or "").strip().lower()
            cur_mode = "replace" if "@replace" in opts or "mode=replace" in opts else "append"
            continue
        if cur_path:
            cur_content.append(line)

    _flush()

    out: List[PatchUpdate] = []
    for rel_path, mode, content in blocks:
        rel_path = rel_path.strip().lstrip("./").replace("\\", "/")
        out.append(PatchUpdate(rel_path=rel_path, mode=mode, content=content))
    return out


def apply_patch_updates(
    skills_patch_text: str,
    rules_patch_text: str,
    workspace_root: Optional[Path] = None,
) -> List[str]:
    """
    应用 skills 与 rules 的补丁。
    返回本次实际写入/更新的文件列表（相对路径）。
    """
    root = workspace_root or Path(__file__).resolve().parents[2]
    updates = parse_patch_updates(skills_patch_text) + parse_patch_updates(rules_patch_text)
    if not updates:
        return []

    changed: List[str] = []

    for u in updates:
        rel = u.rel_path
        if not rel:
            continue
        # 防御：只允许写入工作区内路径（避免路径穿越）
        abs_path = (root / rel).resolve()
        if not str(abs_path).startswith(str(root.resolve())):
            raise RuntimeError(f"非法写入路径：{rel}")

        abs_path.parent.mkdir(parents=True, exist_ok=True)

        if u.mode == "replace":
            new_text = u.content.strip("\n") + "\n"
            abs_path.write_text(new_text, encoding="utf-8")
            changed.append(rel)
            continue

        existing_text = ""
        if abs_path.exists():
            existing_text = abs_path.read_text(encoding="utf-8", errors="ignore")

        existing_paras = _split_paragraphs(existing_text)
        existing_set = set(existing_paras)
        new_paras = _split_paragraphs(u.content)

        merged = existing_paras[:]
        added_any = False
        for p in new_paras:
            if p in existing_set:
                continue
            merged.append(p)
            existing_set.add(p)
            added_any = True

        if not added_any:
            continue

        out_text = "\n\n".join(merged).rstrip("\n") + "\n"
        abs_path.write_text(out_text, encoding="utf-8")
        changed.append(rel)

    # 去重（同一文件可能在多个块更新）
    dedup_changed = []
    seen = set()
    for p in changed:
        if p not in seen:
            seen.add(p)
            dedup_changed.append(p)
    return dedup_changed

