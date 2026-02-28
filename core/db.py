"""SQLite 管理：保存配置（如 provider/模型）和操作记录"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from config import settings


def _get_conn() -> sqlite3.Connection:
    db_path: Path = settings.db_file
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """初始化数据库表（自动迁移新字段）"""
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                provider TEXT DEFAULT 'ollama',
                openai_api_key TEXT,
                openai_base_url TEXT,
                ollama_base_url TEXT DEFAULT 'http://localhost:11434',
                cursor_api_key TEXT,
                cursor_api_base TEXT,
                cursor_repository TEXT,
                cursor_ref TEXT DEFAULT 'main',
                cursor_embedding TEXT DEFAULT 'ollama',
                llm_model TEXT,
                embedding_model TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        # 已有旧表时自动补新字段
        existing_cols = {row[1] for row in cur.execute("PRAGMA table_info(app_settings)")}
        migrations = {
            "provider": "ALTER TABLE app_settings ADD COLUMN provider TEXT DEFAULT 'ollama'",
            "ollama_base_url": "ALTER TABLE app_settings ADD COLUMN ollama_base_url TEXT DEFAULT 'http://localhost:11434'",
            "cursor_api_key": "ALTER TABLE app_settings ADD COLUMN cursor_api_key TEXT",
            "cursor_api_base": "ALTER TABLE app_settings ADD COLUMN cursor_api_base TEXT",
            "cursor_repository": "ALTER TABLE app_settings ADD COLUMN cursor_repository TEXT",
            "cursor_ref": "ALTER TABLE app_settings ADD COLUMN cursor_ref TEXT DEFAULT 'main'",
            "cursor_embedding": "ALTER TABLE app_settings ADD COLUMN cursor_embedding TEXT DEFAULT 'ollama'",
        }
        for col, sql in migrations.items():
            if col not in existing_cols:
                cur.execute(sql)

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS operation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                op_type TEXT NOT NULL,
                collection TEXT,
                file_name TEXT,
                source TEXT,
                extra_json TEXT,
                created_at TEXT DEFAULT (datetime('now','localtime'))
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def load_app_settings() -> Optional[Dict[str, Any]]:
    init_db()
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM app_settings WHERE id = 1")
        row = cur.fetchone()
        if not row:
            return None
        return dict(row)
    finally:
        conn.close()


def save_app_settings(
    provider: str = "ollama",
    openai_api_key: str = "",
    openai_base_url: str = "",
    ollama_base_url: str = "http://localhost:11434",
    cursor_api_key: str = "",
    cursor_api_base: str = "",
    cursor_repository: str = "",
    cursor_ref: str = "main",
    cursor_embedding: str = "ollama",
    llm_model: str = "",
    embedding_model: str = "",
) -> None:
    init_db()
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO app_settings
                (id, provider, openai_api_key, openai_base_url, ollama_base_url,
                 cursor_api_key, cursor_api_base, cursor_repository, cursor_ref, cursor_embedding,
                 llm_model, embedding_model, created_at, updated_at)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now','localtime'), datetime('now','localtime'))
            ON CONFLICT(id) DO UPDATE SET
                provider = excluded.provider,
                openai_api_key = excluded.openai_api_key,
                openai_base_url = excluded.openai_base_url,
                ollama_base_url = excluded.ollama_base_url,
                cursor_api_key = excluded.cursor_api_key,
                cursor_api_base = excluded.cursor_api_base,
                cursor_repository = excluded.cursor_repository,
                cursor_ref = excluded.cursor_ref,
                cursor_embedding = excluded.cursor_embedding,
                llm_model = excluded.llm_model,
                embedding_model = excluded.embedding_model,
                updated_at = datetime('now','localtime')
            """,
            (provider, openai_api_key, openai_base_url, ollama_base_url,
             cursor_api_key, cursor_api_base, cursor_repository, cursor_ref, cursor_embedding,
             llm_model, embedding_model),
        )
        conn.commit()
    finally:
        conn.close()


def add_operation_log(
    op_type: str,
    collection: str,
    file_name: str,
    source: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    init_db()
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO operation_logs (op_type, collection, file_name, source, extra_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                op_type,
                collection or "",
                file_name or "",
                source or "",
                json.dumps(extra or {}, ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()


OP_TYPE_TRAIN_BATCH = "train_batch"
OP_TYPE_TRAIN = "train"
OP_TYPE_TRAIN_ERROR = "train_error"
OP_TYPE_REVIEW_BATCH = "review_batch"
OP_TYPE_REVIEW_ERROR = "review_error"
OP_TYPE_REVIEW_TEXT_ERROR = "review_text_error"


def get_operation_logs(
    op_type: Optional[str] = None,
    collection: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
) -> list:
    init_db()
    conn = _get_conn()
    try:
        cur = conn.cursor()
        sql = """
            SELECT id, op_type, collection, file_name, source, extra_json, created_at
            FROM operation_logs
            WHERE 1=1
        """
        params = []
        if op_type:
            sql += " AND op_type = ?"
            params.append(op_type)
        if collection:
            sql += " AND collection = ?"
            params.append(collection)
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cur.execute(sql, params)
        rows = cur.fetchall()
        result = []
        for row in rows:
            r = dict(row)
            if r.get("extra_json"):
                try:
                    r["extra"] = json.loads(r["extra_json"])
                except Exception:
                    r["extra"] = {}
            else:
                r["extra"] = {}
            result.append(r)
        return result
    finally:
        conn.close()


def get_operation_summary() -> Dict[str, Any]:
    init_db()
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM operation_logs WHERE op_type = ?",
            (OP_TYPE_TRAIN_BATCH,),
        )
        total_train_batches = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM operation_logs WHERE op_type = ?",
            (OP_TYPE_REVIEW_BATCH,),
        )
        total_review_batches = cur.fetchone()[0]
        cur.execute(
            """SELECT COUNT(*) FROM operation_logs
               WHERE op_type IN (?, ?) AND date(created_at) = date('now','localtime')""",
            (OP_TYPE_TRAIN_BATCH, OP_TYPE_REVIEW_BATCH),
        )
        today_ops = cur.fetchone()[0]
        return {
            "total_train_batches": total_train_batches,
            "total_review_batches": total_review_batches,
            "today_operations": today_ops,
        }
    finally:
        conn.close()
