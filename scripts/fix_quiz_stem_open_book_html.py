#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清洗 aicheckword 题库 quiz_questions 题干/解析中的开卷链接 HTML 脏数据。

与 service 下发前清理共用 open_book_stem_sanitize.strip_broken_open_book_html。
对全表逐条比对清洗前后；仅在有变化时 UPDATE（不依赖特征子串预筛）。

用法：
  python scripts/fix_quiz_stem_open_book_html.py --dry-run
  python scripts/fix_quiz_stem_open_book_html.py
  python scripts/fix_quiz_stem_open_book_html.py --collection regulations --include-explanation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core import db  # noqa: E402
from src.core.quiz.open_book_stem_sanitize import strip_broken_open_book_html  # noqa: E402


def _fetch_rows(collection: Optional[str]) -> List[Dict[str, Any]]:
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
            return [dict(r) for r in (cur.fetchall() or [])]
    finally:
        conn.close()


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
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 条有变化的记录（0=不限）")
    args = ap.parse_args()

    collection = str(args.collection or "").strip() or None
    rows = _fetch_rows(collection)

    total = len(rows)
    changed = 0
    changed_stem = 0
    changed_expl = 0
    samples: List[Dict[str, Any]] = []

    for r in rows:
        qid = int(r.get("id") or 0)
        coll = str(r.get("collection") or "regulations")

        stem_before = str(r.get("stem") or "")
        stem_after = strip_broken_open_book_html(stem_before)
        stem_changed = stem_after != stem_before

        expl_after: Optional[str] = None
        expl_changed = False
        if args.include_explanation:
            expl_before = str(r.get("explanation") or "")
            _expl_after = strip_broken_open_book_html(expl_before)
            if _expl_after != expl_before:
                expl_after = _expl_after
                expl_changed = True

        if not stem_changed and not expl_changed:
            continue

        if args.limit and args.limit > 0 and changed >= int(args.limit):
            break

        changed += 1
        if stem_changed:
            changed_stem += 1
        if expl_changed:
            changed_expl += 1

        if len(samples) < 12:
            samples.append(
                {
                    "id": qid,
                    "collection": coll,
                    "stem_before": stem_before[:280],
                    "stem_after": stem_after[:280],
                }
            )

        if not args.dry_run:
            _update_row(
                collection=coll,
                qid=qid,
                stem=stem_after if stem_changed else None,
                explanation=expl_after if expl_changed else None,
            )

    print(
        json.dumps(
            {
                "collection": collection or "(all)",
                "dry_run": bool(args.dry_run),
                "include_explanation": bool(args.include_explanation),
                "rows_scanned": total,
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
