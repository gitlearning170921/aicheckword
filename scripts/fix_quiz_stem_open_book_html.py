#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清洗历史脏数据：quiz_questions 中题干/解析里误写入的前端「开卷查阅」链接 HTML 片段。

背景：早期前端 linkify 逻辑在题干含重复文件名时，会把已生成的
<button data-open-book-file="..."> 标签再次匹配替换，破损后的 HTML 片段
（如 `... 审核点清单-xxx" title="开卷查阅：点击展开全文">《审核点清单-xxx》`）
被存进了 quiz_questions.stem，导致学生端题干直接露出 HTML 代码。

本脚本复用 service._strip_broken_open_book_html 对每条题目的 stem（及可选
explanation）做清洗；仅在清洗后内容发生变化时才 UPDATE。

用法（在 aicheckword 项目根目录、已配置好 DB 与 venv）：
  python scripts/fix_quiz_stem_open_book_html.py --dry-run
  python scripts/fix_quiz_stem_open_book_html.py                 # 落库
  python scripts/fix_quiz_stem_open_book_html.py --collection regulations
  python scripts/fix_quiz_stem_open_book_html.py --include-explanation
  python scripts/fix_quiz_stem_open_book_html.py --limit 50 --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core import db  # noqa: E402
from src.core.quiz.service import _strip_broken_open_book_html  # noqa: E402


# 命中以下任一特征即视为「疑似脏数据」，仅对这些行做清洗与比较，降低全表扫描误改风险
_DIRTY_MARKERS = (
    "exam-open-book-link",
    "data-open-book-file",
    "开卷查阅：点击展开全文",
    "开卷查阅:点击展开全文",
    "<button",
    "</button",
)


def _looks_dirty(text: Optional[str]) -> bool:
    s = str(text or "")
    if not s:
        return False
    return any(marker in s for marker in _DIRTY_MARKERS)


def _fetch_rows(collection: Optional[str], include_explanation: bool) -> List[Dict[str, Any]]:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            if collection:
                cur.execute(
                    "SELECT id, collection, stem, explanation FROM quiz_questions WHERE collection=%s ORDER BY id ASC",
                    (collection,),
                )
            else:
                cur.execute(
                    "SELECT id, collection, stem, explanation FROM quiz_questions ORDER BY collection ASC, id ASC"
                )
            rows = [dict(r) for r in (cur.fetchall() or [])]
    finally:
        conn.close()

    out: List[Dict[str, Any]] = []
    for r in rows:
        dirty = _looks_dirty(r.get("stem")) or (include_explanation and _looks_dirty(r.get("explanation")))
        if dirty:
            out.append(r)
    return out


def _update_row(
    *, collection: str, qid: int, stem: Optional[str], explanation: Optional[str]
) -> None:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        sets: List[str] = []
        params: List[Any] = []
        if stem is not None:
            sets.append("stem=%s")
            params.append(stem)
        if explanation is not None:
            sets.append("explanation=%s")
            params.append(explanation)
        if not sets:
            return
        with conn.cursor() as cur:
            sql = (
                f"UPDATE quiz_questions SET {', '.join(sets)}, updated_at=CURRENT_TIMESTAMP "
                "WHERE collection=%s AND id=%s"
            )
            cur.execute(sql, tuple(params + [collection, int(qid)]))
        conn.commit()
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="", help="仅处理指定 collection；留空=全部")
    ap.add_argument("--dry-run", action="store_true", help="只统计与打印，不落库")
    ap.add_argument("--include-explanation", action="store_true", help="同时清洗 explanation 字段")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 条疑似脏数据（0=不限）")
    args = ap.parse_args()

    collection = str(args.collection or "").strip() or None
    rows = _fetch_rows(collection, include_explanation=bool(args.include_explanation))
    if args.limit and args.limit > 0:
        rows = rows[: int(args.limit)]

    total = len(rows)
    changed = 0
    changed_stem = 0
    changed_expl = 0
    samples: List[Dict[str, Any]] = []

    for r in rows:
        qid = int(r.get("id") or 0)
        coll = str(r.get("collection") or "regulations")

        stem_before = str(r.get("stem") or "")
        stem_after = _strip_broken_open_book_html(stem_before)
        stem_changed = stem_after != stem_before

        expl_after: Optional[str] = None
        expl_changed = False
        if args.include_explanation:
            expl_before = str(r.get("explanation") or "")
            _expl_after = _strip_broken_open_book_html(expl_before)
            if _expl_after != expl_before:
                expl_after = _expl_after
                expl_changed = True

        if not stem_changed and not expl_changed:
            continue

        changed += 1
        if stem_changed:
            changed_stem += 1
        if expl_changed:
            changed_expl += 1

        if len(samples) < 10:
            samples.append(
                {
                    "id": qid,
                    "collection": coll,
                    "stem_before": stem_before[:200],
                    "stem_after": stem_after[:200],
                }
            )

        if not args.dry_run:
            _update_row(
                collection=coll,
                qid=qid,
                stem=stem_after if stem_changed else None,
                explanation=expl_after if expl_changed else None,
            )

    import json

    print(
        json.dumps(
            {
                "collection": collection or "(all)",
                "dry_run": bool(args.dry_run),
                "include_explanation": bool(args.include_explanation),
                "suspect_rows": total,
                "changed": changed,
                "changed_stem": changed_stem,
                "changed_explanation": changed_expl,
                "samples": samples,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
