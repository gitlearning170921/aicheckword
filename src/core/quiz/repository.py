import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.core import db


def _dump(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False)


def _parse_answer_question_id(raw: Any) -> int:
    """将 question_id（含字符串数字或不规范形态）收成正整数；无效返回 0。"""
    if raw is None or isinstance(raw, bool):
        return 0
    if isinstance(raw, int):
        return raw if raw > 0 else 0
    s = str(raw).strip()
    if not s:
        return 0
    try:
        n = int(s)
        return n if n > 0 else 0
    except (TypeError, ValueError):
        pass
    m = re.search(r"\d+", s)
    if not m:
        return 0
    try:
        n = int(m.group(0))
        return n if n > 0 else 0
    except ValueError:
        return 0


def _pick_user_answer_blob(x: Dict[str, Any]) -> Any:
    """兼容 aiword/前端：`answer`、`value` 与 `user_answer` 等价。"""
    if not isinstance(x, dict):
        return None
    for key in ("user_answer", "userAnswer", "answer", "value", "selected", "choice", "response"):
        if key in x and x[key] is not None:
            return x[key]
    return None


def _load(v: Any, default: Any):
    if v is None:
        return default
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except Exception:
        return default


def create_question(
    *,
    collection: str,
    exam_track: str,
    question_hash: str,
    question_type: str,
    difficulty: str,
    category: str,
    knowledge_scope_hash: str,
    stem: str,
    options: List[str],
    answer: Any,
    explanation: str,
    evidence: List[Dict[str, Any]],
    origin: str,
    created_by: str,
) -> int:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO quiz_questions
                (collection, exam_track, question_hash, question_type, difficulty, category, knowledge_scope_hash,
                 stem, options_json, answer_json, explanation, evidence_json, origin, status, version, created_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'active', 1, %s)
                ON DUPLICATE KEY UPDATE
                    question_type=VALUES(question_type),
                    difficulty=VALUES(difficulty),
                    category=VALUES(category),
                    knowledge_scope_hash=VALUES(knowledge_scope_hash),
                    stem=VALUES(stem),
                    options_json=VALUES(options_json),
                    answer_json=VALUES(answer_json),
                    explanation=VALUES(explanation),
                    evidence_json=VALUES(evidence_json),
                    origin=VALUES(origin),
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    collection,
                    exam_track,
                    question_hash,
                    question_type,
                    difficulty,
                    category,
                    knowledge_scope_hash,
                    stem,
                    _dump(options or []),
                    _dump(answer),
                    explanation or "",
                    _dump(evidence or []),
                    origin,
                    created_by or "",
                ),
            )
            if cur.lastrowid:
                qid = int(cur.lastrowid)
            else:
                cur.execute(
                    "SELECT id FROM quiz_questions WHERE collection=%s AND question_hash=%s LIMIT 1",
                    (collection, question_hash),
                )
                row = cur.fetchone() or {}
                qid = int(row.get("id") or 0)
        conn.commit()
        return qid
    finally:
        conn.close()


def upsert_question_bank(
    *,
    collection: str,
    exam_track: str,
    category: str,
    question_type: str,
    difficulty: str,
    knowledge_scope_hash: str,
    question_id: int,
    quality_score: float = 0.0,
) -> None:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO quiz_question_bank
                (collection, exam_track, category, question_type, difficulty, knowledge_scope_hash, question_id, quality_score, is_active, use_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 1, 0)
                ON DUPLICATE KEY UPDATE
                    category=VALUES(category),
                    question_type=VALUES(question_type),
                    difficulty=VALUES(difficulty),
                    quality_score=GREATEST(quality_score, VALUES(quality_score)),
                    is_active=1
                """,
                (
                    collection,
                    exam_track,
                    category or "",
                    question_type,
                    difficulty,
                    knowledge_scope_hash,
                    int(question_id),
                    float(quality_score or 0),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def list_bank_questions(
    *,
    collection: str,
    exam_track: str,
    knowledge_scope_hash: Optional[str] = None,
    question_type: Optional[str] = None,
    difficulty: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 50,
    exclude_question_ids: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    params: List[Any] = [collection, exam_track]
    where = [
        "b.collection=%s",
        "b.exam_track=%s",
        "b.is_active=1",
        "q.id=b.question_id",
        "q.status='active'",
    ]
    if knowledge_scope_hash is not None and str(knowledge_scope_hash).strip():
        where.append("b.knowledge_scope_hash=%s")
        params.append(str(knowledge_scope_hash).strip())
    if question_type:
        where.append("b.question_type=%s")
        params.append(question_type)
    if difficulty:
        where.append("b.difficulty=%s")
        params.append(difficulty)
    if category:
        where.append("b.category=%s")
        params.append(category)
    if exclude_question_ids:
        marks = ",".join(["%s"] * len(exclude_question_ids))
        where.append(f"q.id NOT IN ({marks})")
        params.extend(int(x) for x in exclude_question_ids)
    params.append(max(1, int(limit)))
    sql = f"""
        SELECT q.id, q.question_type, q.difficulty, q.category, q.stem, q.options_json, q.answer_json, q.explanation, q.evidence_json,
               b.knowledge_scope_hash AS knowledge_scope_hash
        FROM quiz_question_bank b, quiz_questions q
        WHERE {' AND '.join(where)}
        ORDER BY b.use_count ASC, b.last_used_at ASC, b.id DESC
        LIMIT %s
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    out = []
    for r in rows:
        r["options"] = _load(r.pop("options_json", "[]"), [])
        r["answer"] = _load(r.pop("answer_json", "null"), None)
        r["evidence"] = _load(r.pop("evidence_json", "[]"), [])
        out.append(r)
    return out


def list_recent_question_stems(
    *, collection: str, exam_track: str, limit: int = 240
) -> List[str]:
    """取最近入库的题干文本，供 AI 录题时与历史题做相似度去重（非精确 hash）。"""
    lim = max(1, min(int(limit or 240), 600))
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT q.stem FROM quiz_questions q
                WHERE q.collection=%s AND q.exam_track=%s
                  AND (q.status IS NULL OR q.status IN ('active', ''))
                ORDER BY q.id DESC
                LIMIT %s
                """,
                (collection, exam_track, lim),
            )
            return [str(r.get("stem") or "") for r in (cur.fetchall() or []) if r]
    finally:
        conn.close()


def list_questions_by_ids(*, collection: str, question_ids: Sequence[int]) -> List[Dict[str, Any]]:
    """按主键批量加载题目（用于练习组卷优先错题/未练，不要求一定在 bank 表）。"""
    ids: List[int] = []
    seen: set[int] = set()
    for raw in question_ids or []:
        try:
            i = int(raw)
        except (TypeError, ValueError):
            continue
        if i <= 0 or i in seen:
            continue
        seen.add(i)
        ids.append(i)
    if not ids:
        return []
    ids = ids[:400]
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    marks = ",".join(["%s"] * len(ids))
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT q.id, q.question_type, q.difficulty, q.category, q.stem,
                       q.options_json, q.answer_json, q.explanation, q.evidence_json,
                       q.knowledge_scope_hash AS knowledge_scope_hash
                FROM quiz_questions q
                WHERE q.collection=%s AND q.id IN ({marks})
                  AND (q.status IS NULL OR q.status IN ('active', ''))
                """,
                (collection, *ids),
            )
            rows = [dict(r) for r in cur.fetchall() or []]
    finally:
        conn.close()
    by_id = {int(r["id"]): r for r in rows if r.get("id") is not None}
    out: List[Dict[str, Any]] = []
    for qid in ids:
        r = by_id.get(int(qid))
        if not r:
            continue
        r = dict(r)
        r["options"] = _load(r.pop("options_json", "[]"), [])
        r["answer"] = _load(r.pop("answer_json", "null"), None)
        r["evidence"] = _load(r.pop("evidence_json", "[]"), [])
        out.append(r)
    return out


def touch_bank_questions(question_ids: Sequence[int]) -> None:
    if not question_ids:
        return
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    marks = ",".join(["%s"] * len(question_ids))
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE quiz_question_bank SET use_count = use_count + 1, last_used_at=CURRENT_TIMESTAMP WHERE question_id IN ({marks})",
                tuple(int(x) for x in question_ids),
            )
        conn.commit()
    finally:
        conn.close()


def create_set(
    *,
    collection: str,
    set_type: str,
    exam_track: str,
    title: str,
    set_config: Dict[str, Any],
    status: str,
    created_by: str,
    items: Sequence[Tuple[int, float]],
) -> int:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO quiz_sets (collection, set_type, exam_track, title, set_config_json, status, created_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (collection, set_type, exam_track, title, _dump(set_config or {}), status, created_by or ""),
            )
            set_id = int(cur.lastrowid)
            if items:
                add_set_items(set_id, items, conn=conn, cur=cur)
        conn.commit()
        return set_id
    finally:
        conn.close()


def add_set_items(
    set_id: int,
    items: Sequence[Tuple[int, float]],
    *,
    replace: bool = False,
    conn=None,
    cur=None,
) -> None:
    """向套题追加题目；replace=True 时先清空该套题下已有条目（同一连接内提交由调用方决定）。"""
    if not items:
        return
    owns_conn = conn is None
    db.init_db()
    conn = conn or db._get_conn()  # noqa: SLF001
    try:
        c = cur or conn.cursor()
        close_cur = cur is None
        try:
            if replace:
                c.execute("DELETE FROM quiz_set_items WHERE set_id=%s", (int(set_id),))
            order_no = 1
            for qid, score in items:
                c.execute(
                    """
                    INSERT INTO quiz_set_items (set_id, question_id, order_no, score)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE order_no=VALUES(order_no), score=VALUES(score)
                    """,
                    (int(set_id), int(qid), order_no, float(score)),
                )
                order_no += 1
        finally:
            if close_cur:
                c.close()
        if owns_conn:
            conn.commit()
    finally:
        if owns_conn:
            conn.close()


def publish_set(set_id: int) -> None:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE quiz_sets SET status='published', updated_at=CURRENT_TIMESTAMP WHERE id=%s", (int(set_id),))
        conn.commit()
    finally:
        conn.close()


def delete_set(set_id: int) -> None:
    """删除套题：软删除（status=deleted）并清空条目，避免外键/历史引用导致异常。"""
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM quiz_set_items WHERE set_id=%s", (int(set_id),))
            cur.execute(
                "UPDATE quiz_sets SET status='deleted', updated_at=CURRENT_TIMESTAMP WHERE id=%s",
                (int(set_id),),
            )
        conn.commit()
    finally:
        conn.close()


def list_sets(
    *,
    collection: str,
    set_type: str = "",
    exam_track: str = "",
    status: str = "",
    q: str = "",
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """套题列表（用于老师端管理与批量操作）。"""
    limit = max(1, int(limit))
    offset = max(0, int(offset))
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        where: List[str] = ["collection=%s"]
        params: List[Any] = [collection]
        if set_type:
            where.append("set_type=%s")
            params.append(set_type)
        if exam_track:
            where.append("exam_track=%s")
            params.append(exam_track)
        if status:
            where.append("status=%s")
            params.append(status)
        else:
            # 默认不展示 deleted
            where.append("status<>'deleted'")
        if q and q.strip():
            where.append("title LIKE %s")
            params.append(f"%{q.strip()}%")

        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) AS n FROM quiz_sets WHERE {' AND '.join(where)}",
                tuple(params),
            )
            total = int((cur.fetchone() or {}).get("n") or 0)
            cur.execute(
                f"""
                SELECT id, collection, set_type, exam_track, title, status, created_by, created_at, updated_at
                FROM quiz_sets
                WHERE {' AND '.join(where)}
                ORDER BY updated_at DESC, id DESC
                LIMIT %s OFFSET %s
                """,
                tuple(params + [limit, offset]),
            )
            rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            rid = r.get("id")
            if rid is not None:
                r.setdefault("set_id", int(rid))
        return rows, total
    finally:
        conn.close()


def load_set(set_id: int) -> Optional[Dict[str, Any]]:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM quiz_sets WHERE id=%s LIMIT 1", (int(set_id),))
            root = cur.fetchone()
            if not root:
                return None
            cur.execute(
                """
                SELECT i.order_no, i.score, q.id AS question_id, q.question_type, q.difficulty, q.category, q.stem,
                       q.options_json, q.answer_json, q.explanation, q.evidence_json
                FROM quiz_set_items i, quiz_questions q
                WHERE i.set_id=%s AND q.id=i.question_id
                ORDER BY i.order_no ASC
                """,
                (int(set_id),),
            )
            items = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    out_items = []
    for r in items:
        r["options"] = _load(r.pop("options_json", "[]"), [])
        r["answer"] = _load(r.pop("answer_json", "null"), None)
        r["evidence"] = _load(r.pop("evidence_json", "[]"), [])
        out_items.append(r)
    d = dict(root)
    d["set_config"] = _load(d.get("set_config_json"), {})
    d.pop("set_config_json", None)
    d["items"] = out_items
    # 对外/前端兼容：路径参数名为 set_id，响应里同时给出 set_id（与行主键 id 相同）
    if d.get("id") is not None:
        d.setdefault("set_id", int(d["id"]))
    return d


def find_recent_set(
    *,
    collection: str,
    set_type: str,
    exam_track: str,
    set_config_hash: str,
) -> Optional[int]:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM quiz_sets
                WHERE collection=%s AND set_type=%s AND exam_track=%s
                  AND JSON_EXTRACT(set_config_json, '$.set_config_hash') = %s
                  AND status IN ('draft', 'published')
                ORDER BY id DESC LIMIT 1
                """,
                (collection, set_type, exam_track, set_config_hash),
            )
            row = cur.fetchone()
            return int(row["id"]) if row else None
    finally:
        conn.close()


def create_attempt(
    *,
    collection: str,
    set_id: int,
    user_id: str,
    mode: str,
    assignment_id: Optional[int] = None,
) -> int:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO quiz_attempts (collection, set_id, assignment_id, user_id, mode, state)
                VALUES (%s, %s, %s, %s, %s, 'in_progress')
                """,
                (collection, int(set_id), assignment_id, user_id or "", mode or "practice"),
            )
            aid = int(cur.lastrowid)
        conn.commit()
        return aid
    finally:
        conn.close()


def get_attempt_by_id(attempt_id: int) -> Optional[Dict[str, Any]]:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, collection, set_id, assignment_id, user_id, mode, state, score_json
                FROM quiz_attempts WHERE id=%s LIMIT 1
                """,
                (int(attempt_id),),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def save_attempt_answers(attempt_id: int, answers: Sequence[Dict[str, Any]]) -> None:
    if not answers:
        return
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            for x in answers:
                if not isinstance(x, dict):
                    continue
                qid = _parse_answer_question_id(x.get("question_id")) or _parse_answer_question_id(x.get("questionId"))
                ua = _pick_user_answer_blob(x)
                if qid <= 0:
                    continue
                cur.execute(
                    """
                    INSERT INTO quiz_answers (attempt_id, question_id, user_answer_json)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        user_answer_json=VALUES(user_answer_json),
                        updated_at=CURRENT_TIMESTAMP
                    """,
                    (int(attempt_id), qid, _dump(ua)),
                )
        conn.commit()
    finally:
        conn.close()


def list_attempt_answers_with_questions(attempt_id: int) -> List[Dict[str, Any]]:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.id AS answer_id, a.attempt_id, a.question_id, a.user_answer_json, a.auto_score, a.final_score, a.is_correct,
                       q.question_type, q.answer_json, q.options_json, q.stem, q.explanation
                FROM quiz_answers a, quiz_questions q
                WHERE a.attempt_id=%s AND q.id=a.question_id
                """,
                (int(attempt_id),),
            )
            rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    out = []
    for r in rows:
        r["user_answer"] = _load(r.pop("user_answer_json"), None)
        r["answer"] = _load(r.pop("answer_json"), None)
        r["options"] = _load(r.pop("options_json"), [])  # list[str]
        out.append(r)
    return out


def update_answer_grade(
    *,
    answer_id: int,
    auto_score: float,
    final_score: float,
    is_correct: bool,
    teacher_comment: str = "",
    graded_by_cache: bool = False,
) -> None:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE quiz_answers
                SET auto_score=%s, final_score=%s, is_correct=%s, teacher_comment=%s, graded_by_cache=%s, updated_at=CURRENT_TIMESTAMP
                WHERE id=%s
                """,
                (float(auto_score), float(final_score), 1 if is_correct else 0, teacher_comment or "", 1 if graded_by_cache else 0, int(answer_id)),
            )
        conn.commit()
    finally:
        conn.close()


def finalize_attempt(attempt_id: int, score_payload: Dict[str, Any], state: str = "submitted") -> None:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE quiz_attempts
                SET state=%s, score_json=%s, submitted_at=CURRENT_TIMESTAMP
                WHERE id=%s
                """,
                (state, _dump(score_payload or {}), int(attempt_id)),
            )
        conn.commit()
    finally:
        conn.close()


def upsert_grading_rule(
    *,
    collection: str,
    paper_id: Optional[int],
    question_id: int,
    version: int,
    answer_key: Any,
    rubric: Any,
    updated_by: str,
) -> None:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO quiz_grading_rules
                (collection, paper_id, question_id, version, answer_key_json, rubric_json, updated_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    answer_key_json=VALUES(answer_key_json),
                    rubric_json=VALUES(rubric_json),
                    updated_by=VALUES(updated_by),
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    collection,
                    paper_id,
                    int(question_id),
                    int(version),
                    _dump(answer_key),
                    _dump(rubric),
                    updated_by or "",
                ),
            )
        conn.commit()
    finally:
        conn.close()


def get_grading_rule(
    *,
    collection: str,
    paper_id: Optional[int],
    question_id: int,
    version: int,
) -> Optional[Dict[str, Any]]:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            if paper_id is None:
                cur.execute(
                    """
                    SELECT * FROM quiz_grading_rules
                    WHERE collection=%s AND paper_id IS NULL AND question_id=%s AND version=%s
                    LIMIT 1
                    """,
                    (collection, int(question_id), int(version)),
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM quiz_grading_rules
                    WHERE collection=%s AND paper_id=%s AND question_id=%s AND version=%s
                    LIMIT 1
                    """,
                    (collection, int(paper_id), int(question_id), int(version)),
                )
            row = cur.fetchone()
            if not row:
                return None
            d = dict(row)
            d["answer_key"] = _load(d.pop("answer_key_json"), None)
            d["rubric"] = _load(d.pop("rubric_json"), None)
            return d
    finally:
        conn.close()


def log_grading_cache_hit(attempt_id: int, question_id: int, hit_type: str, confidence: float) -> None:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO quiz_grading_cache_hits (attempt_id, question_id, hit_type, confidence)
                VALUES (%s, %s, %s, %s)
                """,
                (int(attempt_id), int(question_id), hit_type or "rule", float(confidence)),
            )
        conn.commit()
    finally:
        conn.close()


def create_ingest_job(
    *,
    collection: str,
    exam_track: str,
    target_count: int,
    review_mode: str,
    created_by: str,
) -> int:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO quiz_bank_ingest_jobs
                (collection, exam_track, target_count, generated_count, review_mode, status, created_by)
                VALUES (%s, %s, %s, 0, %s, 'pending', %s)
                """,
                (collection, exam_track, int(target_count), review_mode or "auto_apply", created_by or ""),
            )
            jid = int(cur.lastrowid)
        conn.commit()
        return jid
    finally:
        conn.close()


def update_ingest_job(
    job_id: int,
    *,
    generated_count: int,
    status: str,
    message: str = "",
    set_id: Optional[int] = None,
) -> None:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            if set_id is None:
                cur.execute(
                    """
                    UPDATE quiz_bank_ingest_jobs
                    SET generated_count=%s, status=%s, message=%s, updated_at=CURRENT_TIMESTAMP
                    WHERE id=%s
                    """,
                    (int(generated_count), status, message or "", int(job_id)),
                )
            else:
                cur.execute(
                    """
                    UPDATE quiz_bank_ingest_jobs
                    SET generated_count=%s, status=%s, message=%s, set_id=%s, updated_at=CURRENT_TIMESTAMP
                    WHERE id=%s
                    """,
                    (int(generated_count), status, message or "", int(set_id), int(job_id)),
                )
        conn.commit()
    finally:
        conn.close()


def get_ingest_job(job_id: int) -> Optional[Dict[str, Any]]:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM quiz_bank_ingest_jobs WHERE id=%s LIMIT 1", (int(job_id),))
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def create_review_job(*, collection: str, set_id: int, created_by: str) -> int:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO quiz_set_review_jobs (collection, set_id, status, created_by, message)
                VALUES (%s, %s, 'pending', %s, '')
                """,
                (collection, int(set_id), created_by or ""),
            )
            jid = int(cur.lastrowid)
        conn.commit()
        return jid
    finally:
        conn.close()


def update_review_job(job_id: int, *, status: str, message: str = "", result: Optional[Dict[str, Any]] = None) -> None:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            if result is None:
                cur.execute(
                    """
                    UPDATE quiz_set_review_jobs
                    SET status=%s, message=%s, updated_at=CURRENT_TIMESTAMP
                    WHERE id=%s
                    """,
                    (status, message or "", int(job_id)),
                )
            else:
                cur.execute(
                    """
                    UPDATE quiz_set_review_jobs
                    SET status=%s, message=%s, result_json=%s, updated_at=CURRENT_TIMESTAMP
                    WHERE id=%s
                    """,
                    (status, message or "", _dump(result), int(job_id)),
                )
        conn.commit()
    finally:
        conn.close()


def get_review_job(job_id: int) -> Optional[Dict[str, Any]]:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM quiz_set_review_jobs WHERE id=%s LIMIT 1", (int(job_id),))
            row = cur.fetchone()
            if not row:
                return None
            d = dict(row)
            if d.get("result_json"):
                d["result"] = _load(d.pop("result_json"), {})
            else:
                d.pop("result_json", None)
            return d
    finally:
        conn.close()


def get_bank_tracks_inventory(collection: str) -> List[Dict[str, Any]]:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT exam_track, COUNT(*) AS total
                FROM quiz_question_bank
                WHERE collection=%s AND is_active=1
                GROUP BY exam_track
                ORDER BY exam_track
                """,
                (collection,),
            )
            return [dict(x) for x in cur.fetchall()]
    finally:
        conn.close()


def admin_count_bank_questions(
    *,
    collection: str,
    exam_track: Optional[str] = None,
    q: str = "",
    category: str = "",
    question_type: str = "",
    difficulty: str = "",
    is_active: Optional[bool] = True,
) -> int:
    """教师端题库列表计数（跨 scope）。"""
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            where: List[str] = ["b.collection=%s", "q.id=b.question_id"]
            params: List[Any] = [collection]
            if exam_track:
                where.append("b.exam_track=%s")
                params.append(exam_track)
            if category:
                where.append("b.category=%s")
                params.append(category)
            if question_type:
                where.append("b.question_type=%s")
                params.append(question_type)
            if difficulty:
                where.append("b.difficulty=%s")
                params.append(difficulty)
            if is_active is not None:
                where.append("b.is_active=%s")
                params.append(1 if is_active else 0)
            # 题库层面以 stem 模糊搜索（必要时可扩展 explanation）
            if q and q.strip():
                where.append("q.stem LIKE %s")
                params.append(f"%{q.strip()}%")
            sql = f"""
                SELECT COUNT(DISTINCT q.id) AS n
                FROM quiz_question_bank b, quiz_questions q
                WHERE {' AND '.join(where)}
            """
            cur.execute(sql, tuple(params))
            row = cur.fetchone() or {}
            return int(row.get("n") or 0)
    finally:
        conn.close()


def admin_list_bank_questions(
    *,
    collection: str,
    exam_track: Optional[str] = None,
    q: str = "",
    category: str = "",
    question_type: str = "",
    difficulty: str = "",
    is_active: Optional[bool] = True,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """教师端题库列表（跨 scope；按 question_id 去重）。"""
    limit = max(1, int(limit))
    offset = max(0, int(offset))
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            where: List[str] = ["b.collection=%s", "q.id=b.question_id"]
            params: List[Any] = [collection]
            if exam_track:
                where.append("b.exam_track=%s")
                params.append(exam_track)
            if category:
                where.append("b.category=%s")
                params.append(category)
            if question_type:
                where.append("b.question_type=%s")
                params.append(question_type)
            if difficulty:
                where.append("b.difficulty=%s")
                params.append(difficulty)
            if is_active is not None:
                where.append("b.is_active=%s")
                params.append(1 if is_active else 0)
            if q and q.strip():
                where.append("q.stem LIKE %s")
                params.append(f"%{q.strip()}%")
            params.extend([limit, offset])
            sql = f"""
                SELECT
                    q.id,
                    MAX(b.exam_track) AS exam_track,
                    MAX(b.category) AS category,
                    MAX(b.question_type) AS question_type,
                    MAX(b.difficulty) AS difficulty,
                    MAX(b.is_active) AS is_active,
                    MAX(b.use_count) AS use_count,
                    MAX(b.last_used_at) AS last_used_at,
                    q.stem, q.options_json, q.answer_json, q.explanation, q.evidence_json,
                    q.origin, q.status, q.created_by, q.created_at, q.updated_at
                FROM quiz_question_bank b, quiz_questions q
                WHERE {' AND '.join(where)}
                GROUP BY q.id
                ORDER BY MAX(b.id) DESC
                LIMIT %s OFFSET %s
            """
            cur.execute(sql, tuple(params))
            rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    out: List[Dict[str, Any]] = []
    for r in rows:
        r["options"] = _load(r.pop("options_json", "[]"), [])
        r["answer"] = _load(r.pop("answer_json", "null"), None)
        r["evidence"] = _load(r.pop("evidence_json", "[]"), [])
        r["is_active"] = bool(int(r.get("is_active") or 0))
        out.append(r)
    return out


def admin_update_question(
    *,
    collection: str,
    question_id: int,
    stem: Optional[str] = None,
    options: Optional[List[str]] = None,
    explanation: Optional[str] = None,
    evidence: Optional[List[Dict[str, Any]]] = None,
    status: Optional[str] = None,
) -> None:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        sets: List[str] = []
        params: List[Any] = []
        if stem is not None:
            sets.append("stem=%s")
            params.append(stem)
        if options is not None:
            sets.append("options_json=%s")
            params.append(_dump(options))
        if explanation is not None:
            sets.append("explanation=%s")
            params.append(explanation)
        if evidence is not None:
            sets.append("evidence_json=%s")
            params.append(_dump(evidence))
        if status is not None:
            sets.append("status=%s")
            params.append(status)
        if not sets:
            return
        with conn.cursor() as cur:
            sql = f"UPDATE quiz_questions SET {', '.join(sets)}, updated_at=CURRENT_TIMESTAMP WHERE collection=%s AND id=%s"
            cur.execute(sql, tuple(params + [collection, int(question_id)]))
        conn.commit()
    finally:
        conn.close()


def admin_update_question_answer(*, collection: str, question_id: int, answer: Any) -> None:
    """单独更新答案（允许 answer=None）。"""
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE quiz_questions SET answer_json=%s, updated_at=CURRENT_TIMESTAMP WHERE collection=%s AND id=%s",
                (_dump(answer), collection, int(question_id)),
            )
        conn.commit()
    finally:
        conn.close()


def admin_update_bank_fields(
    *,
    collection: str,
    question_id: int,
    exam_track: Optional[str] = None,
    category: Optional[str] = None,
    question_type: Optional[str] = None,
    difficulty: Optional[str] = None,
    is_active: Optional[bool] = None,
) -> None:
    """更新题库筛选字段：对该 collection 下所有 scope 的 bank 行生效。"""
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        sets: List[str] = []
        params: List[Any] = []
        if exam_track is not None:
            sets.append("exam_track=%s")
            params.append(exam_track)
        if category is not None:
            sets.append("category=%s")
            params.append(category)
        if question_type is not None:
            sets.append("question_type=%s")
            params.append(question_type)
        if difficulty is not None:
            sets.append("difficulty=%s")
            params.append(difficulty)
        if is_active is not None:
            sets.append("is_active=%s")
            params.append(1 if is_active else 0)
        if not sets:
            return
        where = ["collection=%s", "question_id=%s"]
        wparams: List[Any] = [collection, int(question_id)]
        if exam_track is not None:
            # 若要迁移 exam_track，会导致 where 变更；这里保持“更新前的 track”不可知，
            # 因此只在不更新 exam_track 时允许按 track 缩小，否则全 track 更新。
            pass
        with conn.cursor() as cur:
            sql = f"UPDATE quiz_question_bank SET {', '.join(sets)} WHERE {' AND '.join(where)}"
            cur.execute(sql, tuple(params + wparams))
        conn.commit()
    finally:
        conn.close()


def admin_deactivate_question(*, collection: str, question_id: int) -> None:
    """软删除：bank 下架 + question 标为 inactive（不物理删除，避免套题/作答断链）。"""
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE quiz_question_bank SET is_active=0 WHERE collection=%s AND question_id=%s",
                (collection, int(question_id)),
            )
            cur.execute(
                "UPDATE quiz_questions SET status='inactive', updated_at=CURRENT_TIMESTAMP WHERE collection=%s AND id=%s",
                (collection, int(question_id)),
            )
        conn.commit()
    finally:
        conn.close()


def list_wrong_questions_for_student(*, collection: str, user_id: str, limit: int = 80) -> List[Dict[str, Any]]:
    """学生错题：曾在作答中判为错误（排除主观题阅卷中占位）的题目列表。"""
    uid = (user_id or "").strip()
    if not uid:
        return []
    lim = max(1, min(int(limit or 80), 200))
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT q.id AS question_id, q.stem, q.question_type, q.exam_track,
                       q.options_json, q.answer_json,
                       MAX(a.updated_at) AS last_wrong_at, COUNT(*) AS wrong_hits,
                       (
                           SELECT a2.user_answer_json
                           FROM quiz_answers a2
                           INNER JOIN quiz_attempts t2 ON t2.id = a2.attempt_id
                           WHERE a2.question_id = q.id
                             AND t2.collection=%s AND TRIM(t2.user_id)=%s
                             AND a2.is_correct = 0
                             AND (a2.teacher_comment IS NULL OR a2.teacher_comment NOT LIKE %s)
                           ORDER BY a2.updated_at DESC
                           LIMIT 1
                       ) AS user_answer_json
                FROM quiz_answers a
                INNER JOIN quiz_attempts t ON t.id = a.attempt_id
                INNER JOIN quiz_questions q ON q.id = a.question_id
                WHERE t.collection=%s AND TRIM(t.user_id)=%s
                  AND a.is_correct = 0
                  AND (a.teacher_comment IS NULL OR a.teacher_comment NOT LIKE %s)
                GROUP BY q.id, q.stem, q.question_type, q.exam_track, q.options_json, q.answer_json
                ORDER BY MAX(a.updated_at) DESC
                LIMIT %s
                """,
                (
                    collection,
                    uid,
                    "pending_subjective%",
                    collection,
                    uid,
                    "pending_subjective%",
                    lim,
                ),
            )
            out_wrong: List[Dict[str, Any]] = []
            for r in cur.fetchall() or []:
                row = dict(r)
                row["options"] = _load(row.pop("options_json", None), [])
                row["answer"] = _load(row.pop("answer_json", None), None)
                row["user_answer"] = _load(row.pop("user_answer_json", None), None)
                out_wrong.append(row)
            return out_wrong
    finally:
        conn.close()


def count_unpracticed_questions_for_student(
    *, collection: str, user_id: str, exam_track: str = ""
) -> int:
    """学生未练题目总数（不受 list limit 截断）。"""
    uid = (user_id or "").strip()
    if not uid:
        return 0
    track = (exam_track or "").strip()
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            if track:
                cur.execute(
                    """
                    SELECT COUNT(*) AS n
                    FROM quiz_questions q
                    WHERE q.collection=%s
                      AND (q.status IS NULL OR q.status IN ('active', ''))
                      AND q.exam_track=%s
                      AND NOT EXISTS (
                        SELECT 1 FROM quiz_answers a
                        INNER JOIN quiz_attempts t ON t.id = a.attempt_id
                        WHERE a.question_id = q.id AND t.collection=%s AND TRIM(t.user_id)=%s
                      )
                    """,
                    (collection, track, collection, uid),
                )
            else:
                cur.execute(
                    """
                    SELECT COUNT(*) AS n
                    FROM quiz_questions q
                    WHERE q.collection=%s
                      AND (q.status IS NULL OR q.status IN ('active', ''))
                      AND NOT EXISTS (
                        SELECT 1 FROM quiz_answers a
                        INNER JOIN quiz_attempts t ON t.id = a.attempt_id
                        WHERE a.question_id = q.id AND t.collection=%s AND TRIM(t.user_id)=%s
                      )
                    """,
                    (collection, collection, uid),
                )
            row = cur.fetchone()
            return int((row or {}).get("n") or 0)
    finally:
        conn.close()


def list_unpracticed_questions_for_student(
    *, collection: str, user_id: str, exam_track: str = "", limit: int = 100
) -> List[Dict[str, Any]]:
    """学生未练：题库中尚未出现在该用户任何一次作答记录里的题目。"""
    uid = (user_id or "").strip()
    if not uid:
        return []
    lim = max(0, min(int(limit or 100), 300))
    if lim <= 0:
        return []
    track = (exam_track or "").strip()
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            if track:
                cur.execute(
                    """
                    SELECT q.id AS question_id, q.stem, q.question_type, q.exam_track, q.category, q.difficulty
                    FROM quiz_questions q
                    WHERE q.collection=%s
                      AND (q.status IS NULL OR q.status IN ('active', ''))
                      AND q.exam_track=%s
                      AND NOT EXISTS (
                        SELECT 1 FROM quiz_answers a
                        INNER JOIN quiz_attempts t ON t.id = a.attempt_id
                        WHERE a.question_id = q.id AND t.collection=%s AND TRIM(t.user_id)=%s
                      )
                    ORDER BY q.id DESC
                    LIMIT %s
                    """,
                    (collection, track, collection, uid, lim),
                )
            else:
                cur.execute(
                    """
                    SELECT q.id AS question_id, q.stem, q.question_type, q.exam_track, q.category, q.difficulty
                    FROM quiz_questions q
                    WHERE q.collection=%s
                      AND (q.status IS NULL OR q.status IN ('active', ''))
                      AND NOT EXISTS (
                        SELECT 1 FROM quiz_answers a
                        INNER JOIN quiz_attempts t ON t.id = a.attempt_id
                        WHERE a.question_id = q.id AND t.collection=%s AND TRIM(t.user_id)=%s
                      )
                    ORDER BY q.id DESC
                    LIMIT %s
                    """,
                    (collection, collection, uid, lim),
                )
            return [dict(r) for r in cur.fetchall() or []]
    finally:
        conn.close()


def get_stats_options(collection: str) -> Dict[str, Any]:
    """统计页下拉：从作答记录聚合 distinct 学生与 assignment_id（无独立任务表时以 ID 展示）。"""
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    students: List[Dict[str, Any]] = []
    assignments: List[Dict[str, Any]] = []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id AS id, COUNT(*) AS n, MAX(created_at) AS last_at
                FROM quiz_attempts
                WHERE collection=%s AND user_id IS NOT NULL AND TRIM(user_id) <> ''
                GROUP BY user_id
                ORDER BY last_at DESC
                LIMIT 500
                """,
                (collection,),
            )
            for row in cur.fetchall() or []:
                uid = str((row or {}).get("id") or "").strip()
                if not uid:
                    continue
                n = int((row or {}).get("n") or 0)
                students.append({"id": uid, "name": uid if n <= 1 else f"{uid}（{n} 次作答）"})
            cur.execute(
                """
                SELECT assignment_id AS id, COUNT(*) AS n, MAX(created_at) AS last_at
                FROM quiz_attempts
                WHERE collection=%s AND assignment_id IS NOT NULL
                GROUP BY assignment_id
                ORDER BY last_at DESC
                LIMIT 200
                """,
                (collection,),
            )
            for row in cur.fetchall() or []:
                aid = (row or {}).get("id")
                if aid is None:
                    continue
                try:
                    iid = int(aid)
                except (TypeError, ValueError):
                    continue
                n = int((row or {}).get("n") or 0)
                assignments.append(
                    {
                        "id": iid,
                        "name": f"考试任务 #{iid}" if n <= 1 else f"考试任务 #{iid}（{n} 次作答）",
                    }
                )
    finally:
        conn.close()
    return {"students": students, "assignments": assignments}


def get_overview_stats(collection: str) -> Dict[str, Any]:
    db.init_db()
    conn = db._get_conn()  # noqa: SLF001
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM quiz_questions WHERE collection=%s", (collection,))
            questions = int((cur.fetchone() or {}).get("n") or 0)
            cur.execute("SELECT COUNT(*) AS n FROM quiz_sets WHERE collection=%s", (collection,))
            sets_count = int((cur.fetchone() or {}).get("n") or 0)
            cur.execute("SELECT COUNT(*) AS n FROM quiz_attempts WHERE collection=%s", (collection,))
            attempts = int((cur.fetchone() or {}).get("n") or 0)
            cur.execute(
                """
                SELECT
                  SUM(CASE WHEN graded_by_cache=1 THEN 1 ELSE 0 END) AS cache_hits,
                  COUNT(*) AS total
                FROM quiz_answers a, quiz_attempts t
                WHERE t.collection=%s AND a.attempt_id=t.id
                """,
                (collection,),
            )
            row = cur.fetchone() or {}
    finally:
        conn.close()
    total_answers = int(row.get("total") or 0)
    cache_hits = int(row.get("cache_hits") or 0)
    return {
        "questions": questions,
        "sets": sets_count,
        "attempts": attempts,
        "grading_cache_hit_rate": (cache_hits / total_answers) if total_answers else 0.0,
    }

