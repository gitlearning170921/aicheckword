#!/usr/bin/env python3
"""
开工前检查：校验 .cursor/rules/*.mdc 与 .cursor/skills/*/SKILL.md 的元数据与基本结构。
用法（仓库根目录）：
  python scripts/preflight_cursor.py
退出码：0 通过；1 存在错误（含 ERROR）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RULES_DIR = ROOT / ".cursor" / "rules"
SKILLS_DIR = ROOT / ".cursor" / "skills"
SKILL_MAX_LINES = 500
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$|^[a-z0-9]$")


def _parse_simple_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """解析 --- ... --- 的简单 YAML 子集（键值对，无多行值）。"""
    raw = raw.lstrip("\ufeff")
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    if end == -1:
        return {}, raw
    block = raw[3:end].strip()
    rest = raw[end + 4 :].lstrip("\n")
    meta: dict[str, str] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        # 去掉引号
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        meta[k] = v
    return meta, rest


def _check_rules(errors: list[str], warnings: list[str]) -> None:
    if not RULES_DIR.is_dir():
        errors.append(f"ERROR: 缺少目录 {RULES_DIR.relative_to(ROOT)}")
        return
    mdcs = sorted(RULES_DIR.glob("*.mdc"))
    if not mdcs:
        warnings.append("WARN: .cursor/rules 下暂无 .mdc 文件")
        return
    for path in mdcs:
        text = path.read_text(encoding="utf-8", errors="replace")
        meta, body = _parse_simple_frontmatter(text)
        if not meta and not text.strip().startswith("---"):
            errors.append(f"ERROR: {path.relative_to(ROOT)} 缺少 YAML frontmatter（应以 --- 开头）")
            continue
        if "description" not in meta or not meta["description"].strip():
            errors.append(f"ERROR: {path.relative_to(ROOT)} frontmatter 缺少非空 description")
        has_always = "alwaysApply" in meta
        has_globs = "globs" in meta and meta["globs"].strip()
        if not has_always and not has_globs:
            errors.append(
                f"ERROR: {path.relative_to(ROOT)} 需设置 alwaysApply 或 globs（至少其一）"
            )
        if has_always:
            av = meta["alwaysApply"].strip().lower()
            if av not in ("true", "false"):
                errors.append(f"ERROR: {path.relative_to(ROOT)} alwaysApply 应为 true 或 false")
        if not body.strip():
            errors.append(f"ERROR: {path.relative_to(ROOT)} 正文为空")


def _check_skills(errors: list[str], warnings: list[str]) -> None:
    if not SKILLS_DIR.is_dir():
        errors.append(f"ERROR: 缺少目录 {SKILLS_DIR.relative_to(ROOT)}")
        return
    names_seen: dict[str, str] = {}
    skill_dirs = sorted([p for p in SKILLS_DIR.iterdir() if p.is_dir()])
    if not skill_dirs:
        warnings.append("WARN: .cursor/skills 下暂无子目录")
        return
    for d in skill_dirs:
        skill_md = d / "SKILL.md"
        if not skill_md.is_file():
            errors.append(f"ERROR: {d.relative_to(ROOT)} 缺少 SKILL.md")
            continue
        text = skill_md.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        n = len(lines)
        if n > SKILL_MAX_LINES:
            warnings.append(
                f"WARN: {skill_md.relative_to(ROOT)} 共 {n} 行，建议 SKILL.md ≤ {SKILL_MAX_LINES} 行（可拆 reference）"
            )
        meta, body = _parse_simple_frontmatter(text)
        if not meta:
            errors.append(f"ERROR: {skill_md.relative_to(ROOT)} 缺少 YAML frontmatter")
            continue
        name = (meta.get("name") or "").strip()
        desc = (meta.get("description") or "").strip()
        if not name:
            errors.append(f"ERROR: {skill_md.relative_to(ROOT)} frontmatter 缺少 name")
        else:
            if len(name) > 64:
                errors.append(f"ERROR: {skill_md.relative_to(ROOT)} name 长度应 ≤ 64：{name!r}")
            if not NAME_PATTERN.match(name):
                errors.append(
                    f"ERROR: {skill_md.relative_to(ROOT)} name 应为小写字母数字连字符：{name!r}"
                )
            if name in names_seen:
                errors.append(
                    f"ERROR: 重复的 skill name {name!r}：{names_seen[name]} 与 {skill_md}"
                )
            else:
                names_seen[name] = str(skill_md.relative_to(ROOT))
        if not desc:
            errors.append(f"ERROR: {skill_md.relative_to(ROOT)} frontmatter 缺少非空 description")
        if len(desc) > 1024:
            errors.append(f"ERROR: {skill_md.relative_to(ROOT)} description 长度应 ≤ 1024")
        if not body.strip():
            errors.append(f"ERROR: {skill_md.relative_to(ROOT)} 正文为空")


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    print(f"仓库根目录: {ROOT}")
    print("检查 .cursor/rules …")
    _check_rules(errors, warnings)
    print("检查 .cursor/skills …")
    _check_skills(errors, warnings)

    for w in warnings:
        print(w)
    for e in errors:
        print(e)

    if warnings and not errors:
        print("预检完成：有 WARN，无 ERROR。")
    elif not errors:
        print("预检通过：rules/skills 元数据与结构检查 OK。")
    else:
        print("预检失败：请修复上述 ERROR 后重试。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
