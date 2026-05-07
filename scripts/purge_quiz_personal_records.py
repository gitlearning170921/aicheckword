"""
清空题库侧「个人学习/考试」数据，保留题目与套题等通用资产。

会 TRUNCATE（MySQL）以下表（存在才执行）：
  quiz_grading_cache_hits → quiz_answers → quiz_attempts → quiz_wrongbook → quiz_favorites

不删除：quiz_questions、quiz_question_bank、quiz_sets、quiz_set_items、
       quiz_grading_rules、quiz_bank_ingest_jobs、quiz_set_review_jobs

用法（在 aicheckword 项目根目录，需能连上 config 中的 MySQL）:
  python scripts/purge_quiz_personal_records.py --dry-run
  python scripts/purge_quiz_personal_records.py --yes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from src.core import db

# 子表优先（无 FK 声明时也保持习惯顺序）
_TABLES_ORDER = (
    "quiz_grading_cache_hits",
    "quiz_answers",
    "quiz_attempts",
    "quiz_wrongbook",
    "quiz_favorites",
)


def _table_exists(cur, name: str) -> bool:
    cur.execute(
        """
        SELECT 1 AS ok FROM information_schema.TABLES
        WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s LIMIT 1
        """,
        (settings.mysql_database, name),
    )
    return cur.fetchone() is not None


def _count_rows(cur, name: str) -> int:
    cur.execute(f"SELECT COUNT(*) AS n FROM `{name}`")
    row = cur.fetchone() or {}
    return int(row.get("n") or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="清空刷题/考试个人数据（保留题库与套题）")
    parser.add_argument("--dry-run", action="store_true", help="只统计行数，不写库")
    parser.add_argument("--yes", action="store_true", help="跳过交互确认（自动化用）")
    args = parser.parse_args()

    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            present = [t for t in _TABLES_ORDER if _table_exists(cur, t)]
            missing = [t for t in _TABLES_ORDER if t not in present]
            if missing:
                print(f"[info] 以下表不存在或尚未创建，跳过: {', '.join(missing)}")

            counts = {t: _count_rows(cur, t) for t in present}
            for t in present:
                print(f"  {t}: rows={counts[t]}")

            if args.dry_run:
                print("[dry-run] 未修改数据库")
                return

            total = sum(counts.values())
            if total == 0:
                print("[ok] 无数据需清空")
                return

            if not args.yes:
                s = input(f"将清空上述 {len(present)} 张表共约 {total} 行，输入 YES 继续: ")
                if (s or "").strip() != "YES":
                    print("已取消")
                    return

            cur.execute("SET FOREIGN_KEY_CHECKS=0")
            for t in present:
                cur.execute(f"TRUNCATE TABLE `{t}`")
                print(f"[ok] TRUNCATE {t}")
            cur.execute("SET FOREIGN_KEY_CHECKS=1")
        conn.commit()
        print("[done] 个人刷题/考试记录已清空（题库与套题未动）")
    except Exception:
        _rollback = getattr(db, "_rollback_quiet", None)
        if callable(_rollback):
            _rollback(conn)
        else:
            try:
                conn.rollback()
            except Exception:
                pass
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
