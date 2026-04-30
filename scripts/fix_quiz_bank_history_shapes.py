"""
修复历史题库结构问题：
- multiple_choice 多选题：answer 必须为数组，且长度 >= 2（否则属于“伪多选”）
- case_analysis 案例分析：options 必须为空数组；answer 必须为参考作答要点文本（不能是 A/B/C/D）

用法：
  python scripts/fix_quiz_bank_history_shapes.py --collection regulations --dry-run
  python scripts/fix_quiz_bank_history_shapes.py --collection regulations
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Tuple

from src.core import db
from src.core.quiz import repository as repo
from src.core.quiz import service as quiz_service


def _load_json(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    s = str(raw or "").strip()
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def _norm_question_row(q_type: str, options: Any, answer: Any, explanation: str, category: str = "") -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "question_type": q_type,
        "stem": "",
        "options": options,
        "answer": answer,
        "explanation": explanation,
        "category": category,
        "difficulty": "medium",
        "evidence": [],
    }
    return quiz_service._ensure_question_shape(base, fallback_category=category or "")  # noqa: SLF001


def _changed(before_opts: Any, before_ans: Any, after: Dict[str, Any]) -> bool:
    try:
        bo = before_opts if isinstance(before_opts, list) else []
        ao = after.get("options") if isinstance(after.get("options"), list) else []
        if json.dumps(bo, ensure_ascii=False, sort_keys=True) != json.dumps(ao, ensure_ascii=False, sort_keys=True):
            return True
    except Exception:
        pass
    try:
        if json.dumps(before_ans, ensure_ascii=False, sort_keys=True) != json.dumps(after.get("answer"), ensure_ascii=False, sort_keys=True):
            return True
    except Exception:
        return True
    return False


def _fetch_targets(collection: str) -> List[Dict[str, Any]]:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, question_type, options_json, answer_json, explanation
                FROM quiz_questions
                WHERE collection=%s AND question_type IN ('multiple_choice','case_analysis')
                ORDER BY id ASC
                """,
                (collection,),
            )
            return [dict(r) for r in (cur.fetchall() or [])]
    finally:
        conn.close()


def _apply_fix(collection: str, row: Dict[str, Any], dry_run: bool) -> Tuple[bool, Dict[str, Any]]:
    qid = int(row.get("id") or 0)
    qt = str(row.get("question_type") or "").strip().lower()
    before_opts = _load_json(row.get("options_json"), [])
    before_ans = _load_json(row.get("answer_json"), None)
    expl = str(row.get("explanation") or "")
    after = _norm_question_row(qt, before_opts, before_ans, expl)

    if not _changed(before_opts, before_ans, after):
        return False, after

    if dry_run:
        return True, after

    # 写回：options 与 answer 分开更新，避免误覆盖其它字段
    repo.admin_update_question(collection=collection, question_id=qid, options=after.get("options"))
    repo.admin_update_question_answer(collection=collection, question_id=qid, answer=after.get("answer"))
    return True, after


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="regulations")
    ap.add_argument("--dry-run", action="store_true", help="只统计与打印，不落库")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 条（0=不限）")
    args = ap.parse_args()

    rows = _fetch_targets(str(args.collection).strip() or "regulations")
    if args.limit and args.limit > 0:
        rows = rows[: int(args.limit)]

    total = len(rows)
    changed = 0
    changed_mcq = 0
    changed_case = 0
    samples: List[Dict[str, Any]] = []

    for r in rows:
        ok, after = _apply_fix(args.collection, r, dry_run=bool(args.dry_run))
        if not ok:
            continue
        changed += 1
        qt = str(r.get("question_type") or "").strip().lower()
        if qt == "multiple_choice":
            changed_mcq += 1
        elif qt == "case_analysis":
            changed_case += 1
        if len(samples) < 10:
            samples.append(
                {
                    "id": int(r.get("id") or 0),
                    "question_type": qt,
                    "before_answer": _load_json(r.get("answer_json"), None),
                    "after_answer": after.get("answer"),
                    "before_options_len": len(_load_json(r.get("options_json"), []) or []),
                    "after_options_len": len(after.get("options") or []),
                }
            )

    print(
        json.dumps(
            {
                "collection": args.collection,
                "dry_run": bool(args.dry_run),
                "total_targets": total,
                "changed": changed,
                "changed_multiple_choice": changed_mcq,
                "changed_case_analysis": changed_case,
                "samples": samples,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

