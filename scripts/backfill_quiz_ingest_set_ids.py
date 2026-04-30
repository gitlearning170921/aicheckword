#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补全历史 quiz_bank_ingest_jobs.set_id（旧版 ingest 只入库题目、未建套题）。

原理（启发式，非 100% 唯一；并发录题时请先 --dry-run 核对）：
  - 仅处理 status=done 且 generated_count>0 且 set_id IS NULL 的任务行；
  - 在 [created_at, updated_at + 裕量] 时间窗内，按 collection / exam_track / origin=teacher_bulk_ingest
    匹配 quiz_questions，必要时再按 created_by 过滤（任务 created_by 非空时）；
  - 取按 id 升序的前 generated_count 条作为该次任务题目，新建 draft 套题并写 quiz_set_items，
    再 UPDATE 任务的 set_id。

用法（在项目根目录、已配置好 DB 与 venv）：
  python scripts/backfill_quiz_ingest_set_ids.py
  python scripts/backfill_quiz_ingest_set_ids.py --apply
  python scripts/backfill_quiz_ingest_set_ids.py --apply --job-id 12

aiword 侧：补完后若本地「任务记录」仍无 upstream_set_id，可对单条调一次
  GET /api/exam-center/teacher/bank/ingest-jobs/<upstream_job_id>?refresh=1
  或手工在 aiword 库中按 upstream_job_id 更新 upstream_set_id。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core import db  # noqa: E402
from src.core.quiz import repository as repo  # noqa: E402


def _pick_question_ids_for_job(
    row: Dict[str, Any],
    *,
    extra_seconds: int,
) -> Tuple[List[int], str]:
    """返回 (question_ids, note)."""
    collection = row.get("collection") or "regulations"
    exam_track = row.get("exam_track") or "cn"
    need = max(1, int(row.get("generated_count") or 0))
    created_by = (row.get("created_by") or "").strip()

    c0 = row.get("created_at")
    c1 = row.get("updated_at")
    if c0 is None or c1 is None:
        return [], "missing created_at or updated_at"
    hi = c1 + timedelta(seconds=max(0, int(extra_seconds)))

    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            if created_by:
                cur.execute(
                    """
                    SELECT q.id FROM quiz_questions q
                    WHERE q.collection=%s AND q.exam_track=%s
                      AND q.origin=%s
                      AND q.created_by=%s
                      AND q.created_at >= %s AND q.created_at <= %s
                    ORDER BY q.id ASC
                    LIMIT %s
                    """,
                    (
                        collection,
                        exam_track,
                        "teacher_bulk_ingest",
                        created_by,
                        c0,
                        hi,
                        need,
                    ),
                )
            else:
                cur.execute(
                    """
                    SELECT q.id FROM quiz_questions q
                    WHERE q.collection=%s AND q.exam_track=%s
                      AND q.origin=%s
                      AND q.created_at >= %s AND q.created_at <= %s
                    ORDER BY q.id ASC
                    LIMIT %s
                    """,
                    (collection, exam_track, "teacher_bulk_ingest", c0, hi, need),
                )
            out = [int(r["id"]) for r in cur.fetchall()]
    finally:
        conn.close()

    if len(out) < need:
        return out, f"matched {len(out)}/{need} (time window [{c0}, {hi}], created_by={created_by!r})"
    return out, f"matched {len(out)}/{need}"


def backfill(
    *,
    apply: bool,
    job_id: Optional[int],
    extra_seconds: int,
) -> int:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            if job_id is not None:
                cur.execute(
                    """
                    SELECT * FROM quiz_bank_ingest_jobs
                    WHERE id=%s AND status='done' AND generated_count>0
                      AND (set_id IS NULL OR set_id=0)
                    """,
                    (int(job_id),),
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM quiz_bank_ingest_jobs
                    WHERE status='done' AND generated_count>0
                      AND (set_id IS NULL OR set_id=0)
                    ORDER BY id ASC
                    """
                )
            jobs = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    if not jobs:
        print("[INFO] 没有需要补全的任务（或 --job-id 不存在/不满足条件）。")
        return 0

    n_ok = 0
    for row in jobs:
        jid = int(row["id"])
        qids, note = _pick_question_ids_for_job(row, extra_seconds=extra_seconds)
        need = max(1, int(row.get("generated_count") or 0))
        title = f"历史补全-AI录题-job{jid}"
        print(f"\n--- job id={jid} exam_track={row.get('exam_track')} gen={need} ---")
        print(f"    {note}")
        print(f"    question_ids={qids}")

        if len(qids) != need:
            print(f"    [SKIP] 题目数量与 generated_count 不一致，请人工核对时间窗或并发任务。")
            continue

        if not apply:
            print("    [DRY-RUN] 将创建套题并写回 set_id（加 --apply 执行）。")
            continue

        set_id = repo.create_set(
            collection=row.get("collection") or "regulations",
            set_type="bank_ingest",
            exam_track=row.get("exam_track") or "cn",
            title=title,
            set_config={
                "source": "backfill_quiz_ingest_set_ids",
                "ingest_job_id": jid,
            },
            status="draft",
            created_by=(row.get("created_by") or "") or "backfill",
            items=[],
        )
        repo.add_set_items(set_id, [(qid, 1.0) for qid in qids], replace=True)
        msg = (row.get("message") or "").strip() or "ok"
        repo.update_ingest_job(
            jid,
            generated_count=need,
            status="done",
            message=msg + " [set_id backfilled]",
            set_id=set_id,
        )
        print(f"    [OK] set_id={set_id}")
        n_ok += 1

    if apply:
        print(f"\n[DONE] 成功补全 {n_ok} 条任务。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="实际写入数据库；省略则只打印计划不修改")
    ap.add_argument("--job-id", type=int, default=None, help="只处理指定任务 id")
    ap.add_argument(
        "--extra-seconds",
        type=int,
        default=600,
        help="在任务 updated_at 之后再扩展的时间窗（秒），默认 600",
    )
    args = ap.parse_args()
    apply = bool(args.apply)
    return backfill(apply=apply, job_id=args.job_id, extra_seconds=max(0, int(args.extra_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
