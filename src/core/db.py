"""MySQL 管理：保存配置（provider/模型）和所有操作记录，含使用的模型信息"""

import json
from typing import Any, Dict, Optional

import pymysql
from pymysql.cursors import DictCursor

from config import settings


def _get_conn():
    """获取 MySQL 连接（使用指定数据库）"""
    return pymysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
        charset=settings.mysql_charset,
        cursorclass=DictCursor,
    )


def _ensure_database():
    """确保数据库存在"""
    conn = pymysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        charset=settings.mysql_charset,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE DATABASE IF NOT EXISTS `%s` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                % settings.mysql_database.replace("`", "``")
            )
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """初始化数据库与表结构"""
    _ensure_database()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS app_settings (
                    id INT PRIMARY KEY DEFAULT 1,
                    provider VARCHAR(32) DEFAULT 'ollama',
                    openai_api_key VARCHAR(1024),
                    openai_base_url VARCHAR(512),
                    ollama_base_url VARCHAR(256) DEFAULT 'http://localhost:11434',
                    cursor_api_key VARCHAR(512),
                    cursor_api_base VARCHAR(512),
                    cursor_repository VARCHAR(512),
                    cursor_ref VARCHAR(64) DEFAULT 'main',
                    cursor_embedding VARCHAR(32) DEFAULT 'ollama',
                    llm_model VARCHAR(128),
                    embedding_model VARCHAR(128),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS operation_logs (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    op_type VARCHAR(64) NOT NULL,
                    collection VARCHAR(128) DEFAULT '',
                    file_name VARCHAR(512) DEFAULT '',
                    source VARCHAR(1024) DEFAULT '',
                    extra_json LONGTEXT,
                    model_info VARCHAR(256) DEFAULT '',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_op_type (op_type),
                    INDEX idx_collection (collection),
                    INDEX idx_created_at (created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS audit_reports (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    collection VARCHAR(128) DEFAULT '',
                    file_name VARCHAR(512),
                    report_json LONGTEXT NOT NULL,
                    model_info VARCHAR(256) DEFAULT '',
                    total_points INT DEFAULT 0,
                    high_count INT DEFAULT 0,
                    medium_count INT DEFAULT 0,
                    low_count INT DEFAULT 0,
                    info_count INT DEFAULT 0,
                    summary TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_ar_collection (collection),
                    INDEX idx_ar_created (created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_docs (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    collection VARCHAR(128) NOT NULL,
                    file_name VARCHAR(512),
                    chunk_index INT DEFAULT 0,
                    content LONGTEXT NOT NULL,
                    metadata_json LONGTEXT,
                    category VARCHAR(32) DEFAULT 'regulation',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_kd_collection (collection),
                    INDEX idx_kd_file (file_name),
                    INDEX idx_kd_category (category)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS audit_corrections (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    report_id BIGINT NOT NULL,
                    point_index INT NOT NULL,
                    collection VARCHAR(128) DEFAULT '',
                    file_name VARCHAR(512) DEFAULT '',
                    original_json LONGTEXT,
                    corrected_json LONGTEXT NOT NULL,
                    fed_to_kb TINYINT DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_ac_report (report_id),
                    INDEX idx_ac_collection (collection)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS audit_checklists (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    collection VARCHAR(128) DEFAULT '',
                    name VARCHAR(256) NOT NULL DEFAULT '',
                    checklist_json LONGTEXT NOT NULL,
                    total_points INT DEFAULT 0,
                    base_file VARCHAR(512) DEFAULT '',
                    model_info VARCHAR(256) DEFAULT '',
                    status VARCHAR(32) DEFAULT 'draft',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_cl_collection (collection),
                    INDEX idx_cl_status (status),
                    INDEX idx_cl_created (created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS dimension_options (
                    id INT PRIMARY KEY DEFAULT 1,
                    registration_countries LONGTEXT COMMENT 'JSON array, default ["中国","美国","欧盟"]',
                    project_forms LONGTEXT COMMENT 'JSON array, default ["Web","APP","PC"]',
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    collection VARCHAR(128) NOT NULL DEFAULT 'regulations',
                    name VARCHAR(256) NOT NULL,
                    registration_country VARCHAR(128) NOT NULL DEFAULT '',
                    registration_type VARCHAR(128) NOT NULL DEFAULT '',
                    registration_component VARCHAR(128) NOT NULL DEFAULT '',
                    project_form VARCHAR(128) NOT NULL DEFAULT '',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_proj_collection (collection)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS project_knowledge_docs (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    project_id BIGINT NOT NULL,
                    collection VARCHAR(128) NOT NULL DEFAULT '',
                    file_name VARCHAR(512) DEFAULT '',
                    chunk_index INT DEFAULT 0,
                    content LONGTEXT NOT NULL,
                    metadata_json LONGTEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_pkd_project (project_id),
                    INDEX idx_pkd_collection (collection)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS checkpoint_docs (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    collection VARCHAR(128) NOT NULL DEFAULT '',
                    file_name VARCHAR(512) DEFAULT '',
                    chunk_index INT DEFAULT 0,
                    content LONGTEXT NOT NULL,
                    metadata_json LONGTEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_cd_collection (collection),
                    INDEX idx_cd_file (file_name)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS project_cases (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    collection VARCHAR(128) NOT NULL DEFAULT 'regulations',
                    case_name VARCHAR(256) NOT NULL DEFAULT '',
                    product_name VARCHAR(512) DEFAULT '',
                    registration_country VARCHAR(128) DEFAULT '',
                    registration_type VARCHAR(128) DEFAULT '',
                    registration_component VARCHAR(128) DEFAULT '',
                    project_form VARCHAR(128) DEFAULT '',
                    scope_of_application TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_pc_collection (collection)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            _add_column_if_missing(cur, "project_cases", "case_name_en", "VARCHAR(512) DEFAULT '' COMMENT '案例名称英文'")
            _add_column_if_missing(cur, "project_cases", "product_name_en", "VARCHAR(512) DEFAULT '' COMMENT '产品名称英文'")
            _add_column_if_missing(cur, "project_cases", "registration_country_en", "VARCHAR(256) DEFAULT '' COMMENT '注册国家英文'")
            _add_column_if_missing(cur, "project_cases", "document_language", "VARCHAR(32) DEFAULT 'zh' COMMENT '案例文档语言：zh中文版/en英文版/both中英文'")
            _add_column_if_missing(cur, "project_cases", "project_key", "VARCHAR(256) DEFAULT '' COMMENT '关联项目标识：同一项目下多语言/多国家案例填相同值'")
            _add_column_if_missing(cur, "operation_logs", "model_info", "VARCHAR(256) DEFAULT ''")
            _add_column_if_missing(cur, "knowledge_docs", "category", "VARCHAR(32) DEFAULT 'regulation'")
            _add_column_if_missing(cur, "knowledge_docs", "case_id", "BIGINT DEFAULT NULL COMMENT '关联 project_cases.id，仅 category=project_case 时有值'")
            _add_column_if_missing(cur, "app_settings", "cursor_verify_ssl", "TINYINT(1) DEFAULT 1")
            _add_column_if_missing(cur, "app_settings", "cursor_trust_env", "TINYINT(1) DEFAULT 1")
            _add_column_if_missing(cur, "projects", "basic_info_text", "TEXT COMMENT '从项目资料中提取的基本信息，审核时与待审文档一致性核对'")
            _add_column_if_missing(cur, "projects", "system_functionality_text", "TEXT COMMENT '从安装包或URL识别的系统功能描述，审核时与待审文档一致性核对'")
            _add_column_if_missing(cur, "projects", "system_functionality_source", "VARCHAR(64) DEFAULT '' COMMENT 'package|url|空'")
            _add_column_if_missing(cur, "projects", "scope_of_application", "TEXT COMMENT '产品适用范围，审核时要求文档描述内容不超出此范围'")
            _add_column_if_missing(cur, "projects", "product_name", "VARCHAR(512) DEFAULT '' COMMENT '产品名称，与项目名称一并加入审核点/一致性核对'")
            _add_column_if_missing(cur, "projects", "name_en", "VARCHAR(256) DEFAULT '' COMMENT '项目名称英文'")
            _add_column_if_missing(cur, "projects", "product_name_en", "VARCHAR(512) DEFAULT '' COMMENT '产品名称英文'")
            _add_column_if_missing(cur, "projects", "registration_country_en", "VARCHAR(128) DEFAULT '' COMMENT '注册国家英文'")
            _add_column_if_missing(cur, "app_settings", "review_extra_instructions", "LONGTEXT COMMENT '自定义审核要求/提示词，会追加到审核上下文中'")
            _add_column_if_missing(cur, "app_settings", "review_system_prompt", "LONGTEXT COMMENT '审核系统提示词，为空则使用内置默认'")
            _add_column_if_missing(cur, "app_settings", "review_user_prompt", "LONGTEXT COMMENT '审核用户提示词模板，支持 {context} {file_name} {document_content}，为空则使用内置默认'")
            _add_column_if_missing(cur, "app_settings", "checklist_generate_prompt", "LONGTEXT COMMENT '生成审核点提示词，为空则使用内置默认'")
            _add_column_if_missing(cur, "app_settings", "checklist_optimize_prompt", "LONGTEXT COMMENT '优化审核点提示词，为空则使用内置默认'")
            _add_column_if_missing(cur, "audit_checklists", "document_language", "VARCHAR(32) DEFAULT '' COMMENT '审核点适用文档语言：zh/en/both，空表示不限定'")
            _add_column_if_missing(cur, "app_settings", "project_basic_info_prompt", "LONGTEXT COMMENT '项目基本信息提取提示词，为空则使用内置默认'")
            _add_column_if_missing(cur, "app_settings", "review_summary_prompt", "LONGTEXT COMMENT '审核总结提示词，为空则使用内置默认'")
            _add_column_if_missing(cur, "dimension_options", "country_extra_keywords", "LONGTEXT COMMENT 'JSON: 国家(以页面选择为准)->法规关键词，用于知识库检索与扩大审核面，如 {\"欧盟\":[\"MDR\"]}'")
            _init_dimension_options(cur)
        conn.commit()
    finally:
        conn.close()


def _add_column_if_missing(cur, table: str, column: str, definition: str):
    try:
        cur.execute("""
            SELECT COUNT(*) AS n FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s
        """, (settings.mysql_database, table, column))
        if cur.fetchone()["n"] == 0:
            cur.execute(f"ALTER TABLE `{table}` ADD COLUMN `{column}` {definition}")
    except Exception:
        pass


def _init_dimension_options(cur):
    """初始化维度选项默认值（注册国家、项目形态）"""
    try:
        cur.execute("SELECT id FROM dimension_options WHERE id = 1")
        if cur.fetchone():
            return
        cur.execute("""
            INSERT INTO dimension_options (id, registration_countries, project_forms)
            VALUES (1, %s, %s)
        """, (
            json.dumps(["中国", "美国", "欧盟"], ensure_ascii=False),
            json.dumps(["Web", "APP", "PC"], ensure_ascii=False),
        ))
    except Exception:
        pass


def load_app_settings() -> Optional[Dict[str, Any]]:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM app_settings WHERE id = 1")
            row = cur.fetchone()
        if not row:
            return None
        return dict(row)
    finally:
        conn.close()


def get_review_extra_instructions() -> str:
    """获取自定义审核要求/提示词（会追加到审核上下文），用于提升审核质量。"""
    row = load_app_settings()
    if not row:
        return ""
    return (row.get("review_extra_instructions") or "").strip()


def update_review_extra_instructions(instructions: str) -> None:
    """保存自定义审核要求/提示词。"""
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE app_settings SET review_extra_instructions = %s, updated_at = CURRENT_TIMESTAMP WHERE id = 1",
                (instructions or "",),
            )
            if cur.rowcount == 0:
                cur.execute(
                    "INSERT INTO app_settings (id, review_extra_instructions) VALUES (1, %s)",
                    (instructions or "",),
                )
        conn.commit()
    finally:
        conn.close()


def get_review_system_prompt() -> Optional[str]:
    """获取数据库中保存的审核系统提示词，空或 None 表示使用内置默认。"""
    row = load_app_settings()
    if not row:
        return None
    s = (row.get("review_system_prompt") or "").strip()
    return s if s else None


def get_review_user_prompt() -> Optional[str]:
    """获取数据库中保存的审核用户提示词模板（支持 {context} {file_name} {document_content}），空或 None 表示使用内置默认。"""
    row = load_app_settings()
    if not row:
        return None
    s = (row.get("review_user_prompt") or "").strip()
    return s if s else None


def update_review_prompts(system_prompt: Optional[str] = None, user_prompt: Optional[str] = None) -> None:
    """保存审核系统/用户提示词到数据库。空字符串表示清空（后续使用内置默认）。"""
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            updates = []
            params = []
            if system_prompt is not None:
                updates.append("review_system_prompt = %s")
                params.append(system_prompt or "")
            if user_prompt is not None:
                updates.append("review_user_prompt = %s")
                params.append(user_prompt or "")
            if updates:
                params.append(1)
                cur.execute(
                    "UPDATE app_settings SET " + ", ".join(updates) + ", updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                    params,
                )
        conn.commit()
    finally:
        conn.close()


def get_prompt_by_key(key: str) -> Optional[str]:
    """根据功能模块 key 获取已保存的提示词，空或 None 表示使用内置默认。key 见 PROMPT_KEYS。"""
    row = load_app_settings()
    if not row:
        return None
    val = (row.get(key) or "").strip()
    return val if val else None


def update_prompt_by_key(key: str, content: Optional[str]) -> None:
    """保存指定 key 的提示词到数据库。空字符串表示清空（使用内置默认）。"""
    if key not in PROMPT_KEYS:
        return
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE app_settings SET `" + key + "` = %s, updated_at = CURRENT_TIMESTAMP WHERE id = 1",
                (content or "",),
            )
        conn.commit()
    finally:
        conn.close()


# 提示词功能模块与 app_settings 列名对应（用于分模块配置与入库）
PROMPT_KEYS = [
    "review_system_prompt",
    "review_user_prompt",
    "review_extra_instructions",
    "checklist_generate_prompt",
    "checklist_optimize_prompt",
    "project_basic_info_prompt",
    "review_summary_prompt",
]


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
    cursor_verify_ssl: bool = True,
    cursor_trust_env: bool = True,
    llm_model: str = "",
    embedding_model: str = "",
) -> None:
    init_db()
    conn = _get_conn()
    try:
        verify_ssl_int = 1 if cursor_verify_ssl else 0
        trust_env_int = 1 if cursor_trust_env else 0
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO app_settings (
                    id, provider, openai_api_key, openai_base_url, ollama_base_url,
                    cursor_api_key, cursor_api_base, cursor_repository, cursor_ref, cursor_embedding,
                    cursor_verify_ssl, cursor_trust_env, llm_model, embedding_model
                ) VALUES (
                    1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) ON DUPLICATE KEY UPDATE
                    provider = VALUES(provider),
                    openai_api_key = VALUES(openai_api_key),
                    openai_base_url = VALUES(openai_base_url),
                    ollama_base_url = VALUES(ollama_base_url),
                    cursor_api_key = VALUES(cursor_api_key),
                    cursor_api_base = VALUES(cursor_api_base),
                    cursor_repository = VALUES(cursor_repository),
                    cursor_ref = VALUES(cursor_ref),
                    cursor_embedding = VALUES(cursor_embedding),
                    cursor_verify_ssl = VALUES(cursor_verify_ssl),
                    cursor_trust_env = VALUES(cursor_trust_env),
                    llm_model = VALUES(llm_model),
                    embedding_model = VALUES(embedding_model),
                    updated_at = CURRENT_TIMESTAMP
            """, (
                provider, openai_api_key, openai_base_url, ollama_base_url,
                cursor_api_key, cursor_api_base, cursor_repository, cursor_ref, cursor_embedding,
                verify_ssl_int, trust_env_int, llm_model, embedding_model,
            ))
        conn.commit()
    finally:
        conn.close()


def add_operation_log(
    op_type: str,
    collection: str,
    file_name: str,
    source: str = "",
    extra: Optional[Dict[str, Any]] = None,
    model_info: str = "",
) -> None:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO operation_logs (op_type, collection, file_name, source, extra_json, model_info)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                op_type,
                collection or "",
                file_name or "",
                source or "",
                json.dumps(extra or {}, ensure_ascii=False),
                model_info or "",
            ))
        conn.commit()
    finally:
        conn.close()


OP_TYPE_TRAIN_BATCH = "train_batch"
OP_TYPE_TRAIN = "train"
OP_TYPE_TRAIN_ERROR = "train_error"
OP_TYPE_REVIEW_BATCH = "review_batch"
OP_TYPE_REVIEW = "review"
OP_TYPE_REVIEW_ERROR = "review_error"
OP_TYPE_REVIEW_TEXT = "review_text"
OP_TYPE_REVIEW_TEXT_ERROR = "review_text_error"
OP_TYPE_CORRECTION = "correction"
OP_TYPE_GENERATE_CHECKLIST = "generate_checklist"
OP_TYPE_TRAIN_CHECKLIST = "train_checklist"
OP_TYPE_TRAIN_PROJECT = "train_project"
OP_TYPE_TRAIN_PROJECT_ERROR = "train_project_error"  # 单文件失败或整批中断


def get_current_model_info() -> str:
    from config import settings
    p = (settings.provider or "").lower()
    llm = settings.llm_model or ""
    emb = settings.embedding_model or ""
    if p == "ollama":
        return f"Ollama / LLM:{llm} / Embed:{emb}"
    if p == "cursor":
        return f"Cursor Agent / Embed:{emb}"
    return f"OpenAI / LLM:{llm} / Embed:{emb}"


def get_operation_logs(
    op_type: Optional[str] = None,
    collection: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
) -> list:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            sql = """
                SELECT id, op_type, collection, file_name, source, extra_json, model_info, created_at
                FROM operation_logs
                WHERE 1=1
            """
            params = []
            if op_type:
                sql += " AND op_type = %s"
                params.append(op_type)
            if collection:
                sql += " AND collection = %s"
                params.append(collection)
            sql += " ORDER BY id DESC LIMIT %s OFFSET %s"
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
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM operation_logs WHERE op_type = %s",
                (OP_TYPE_TRAIN_BATCH,),
            )
            total_train_batches = cur.fetchone()["c"]
            cur.execute(
                "SELECT COUNT(*) AS c FROM operation_logs WHERE op_type = %s",
                (OP_TYPE_REVIEW_BATCH,),
            )
            total_review_batches = cur.fetchone()["c"]
            cur.execute("""
                SELECT COUNT(*) AS c FROM operation_logs
                WHERE op_type IN (%s, %s) AND DATE(created_at) = CURDATE()
            """, (OP_TYPE_TRAIN_BATCH, OP_TYPE_REVIEW_BATCH))
            today_ops = cur.fetchone()["c"]
        return {
            "total_train_batches": total_train_batches,
            "total_review_batches": total_review_batches,
            "today_operations": today_ops,
        }
    finally:
        conn.close()


def save_audit_report(
    collection: str,
    report: Dict[str, Any],
    model_info: str = "",
) -> int:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            summary = report.get("summary")
            if summary is None:
                summary = ""
            summary = str(summary)[:65535]
            report_for_json = {k: v for k, v in report.items() if k not in ("_original_path", "_kdocs_download_url")}
            report_json_str = json.dumps(report_for_json, ensure_ascii=False)
            cur.execute("""
                INSERT INTO audit_reports
                    (collection, file_name, report_json, model_info,
                     total_points, high_count, medium_count, low_count, info_count, summary)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                collection or "",
                (report.get("file_name") or "")[:512],
                report_json_str,
                (model_info or "")[:256],
                report.get("total_points", 0),
                report.get("high_count", 0),
                report.get("medium_count", 0),
                report.get("low_count", 0),
                report.get("info_count", 0),
                summary,
            ))
            report_id = cur.lastrowid
        conn.commit()
        return report_id
    finally:
        conn.close()


def get_audit_reports(
    collection: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            sql = "SELECT * FROM audit_reports WHERE 1=1"
            params = []
            if collection:
                sql += " AND collection = %s"
                params.append(collection)
            sql += " ORDER BY id DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            cur.execute(sql, params)
            rows = cur.fetchall()
        result = []
        for row in rows:
            r = dict(row)
            if r.get("report_json"):
                try:
                    r["report"] = json.loads(r["report_json"])
                except Exception:
                    r["report"] = {}
            else:
                r["report"] = {}
            result.append(r)
        return result
    finally:
        conn.close()


def get_audit_report_by_id(report_id: int) -> Optional[Dict[str, Any]]:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM audit_reports WHERE id = %s", (report_id,))
            row = cur.fetchone()
        if not row:
            return None
        r = dict(row)
        r["report"] = json.loads(r.get("report_json") or "{}") if r.get("report_json") else {}
        return r
    finally:
        conn.close()


def get_audit_reports_by_file_name(
    collection: str,
    file_name: str,
    limit: int = 100,
) -> list:
    """按文件名查询该文件的所有历史审核报告（用于整合多份报告）。"""
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM audit_reports
                WHERE collection = %s AND file_name = %s
                ORDER BY id DESC LIMIT %s
            """, (collection, file_name, limit))
            rows = cur.fetchall()
        result = []
        for row in rows:
            r = dict(row)
            if r.get("report_json"):
                try:
                    r["report"] = json.loads(r["report_json"])
                except Exception:
                    r["report"] = {}
            else:
                r["report"] = {}
            result.append(r)
        return result
    finally:
        conn.close()


def get_audit_report_file_names(collection: str, limit: int = 200) -> list:
    """返回当前 collection 下有过审核报告的不重复文件名列表（用于按文件整合）。"""
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT file_name FROM audit_reports
                WHERE collection = %s AND file_name IS NOT NULL AND file_name != ''
                ORDER BY file_name LIMIT %s
            """, (collection, limit))
            rows = cur.fetchall()
        return [r["file_name"] for r in rows]
    finally:
        conn.close()


def save_knowledge_docs(
    collection: str,
    file_name: str,
    chunks: list,
    category: str = "regulation",
) -> int:
    init_db()
    if not chunks:
        return 0
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            rows = []
            for idx, doc in enumerate(chunks):
                content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                meta = doc.metadata if hasattr(doc, "metadata") else {}
                rows.append((
                    collection, file_name, idx,
                    content,
                    json.dumps(meta, ensure_ascii=False),
                    category,
                ))
            cur.executemany("""
                INSERT INTO knowledge_docs (collection, file_name, chunk_index, content, metadata_json, category)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, rows)
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def get_knowledge_docs(
    collection: Optional[str] = None,
    file_name: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
) -> list:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            sql = "SELECT id, collection, file_name, chunk_index, content, metadata_json, category, created_at FROM knowledge_docs WHERE 1=1"
            params = []
            if collection:
                sql += " AND collection = %s"
                params.append(collection)
            if file_name:
                sql += " AND file_name = %s"
                params.append(file_name)
            if category:
                sql += " AND category = %s"
                params.append(category)
            sql += " ORDER BY id DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_knowledge_stats_by_category(collection: Optional[str] = None) -> Dict[str, Any]:
    """按分类统计：总文件数/块数 + 各分类(regulation/program/project_case)的文件数/块数"""
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            where = " WHERE collection = %s" if collection else ""
            params = [collection] if collection else []

            cur.execute(
                "SELECT COUNT(*) AS total_chunks, COUNT(DISTINCT file_name) AS total_files FROM knowledge_docs" + where,
                params if params else (),
            )
            row = cur.fetchone()
            total_chunks = row["total_chunks"] or 0
            total_files = row["total_files"] or 0

            cur.execute(
                """
                SELECT category,
                       COUNT(*) AS chunks,
                       COUNT(DISTINCT file_name) AS files
                FROM knowledge_docs""" + where + """
                GROUP BY category
                """,
                params if params else (),
            )
            rows = cur.fetchall()
        by_category = {}
        for r in rows:
            cat = r.get("category") or "regulation"
            by_category[cat] = {"chunks": r["chunks"], "files": r["files"]}
        for cat in ("regulation", "program", "project_case", "glossary"):
            if cat not in by_category:
                by_category[cat] = {"chunks": 0, "files": 0}
        return {
            "total_chunks": total_chunks,
            "total_files": total_files,
            "by_category": by_category,
        }
    finally:
        conn.close()


def get_knowledge_stats(collection: Optional[str] = None) -> Dict[str, Any]:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            if collection:
                cur.execute(
                    "SELECT COUNT(*) AS total_chunks, COUNT(DISTINCT file_name) AS total_files FROM knowledge_docs WHERE collection = %s",
                    (collection,),
                )
            else:
                cur.execute("SELECT COUNT(*) AS total_chunks, COUNT(DISTINCT file_name) AS total_files FROM knowledge_docs")
            return dict(cur.fetchone())
    finally:
        conn.close()


def clear_knowledge_docs(collection: str) -> int:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM knowledge_docs WHERE collection = %s", (collection,))
            deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


def get_existing_file_names(
    collection: str,
    category: Optional[str] = None,
    case_id: Optional[int] = None,
) -> list:
    """返回该知识库下已存在的文件名列表（用于检测重复）。
    当 category='project_case' 且 case_id 有值时，仅返回该案例下已存在的文件名（按 项目/产品/语言/国家/类别/组成+文档名 组合判断重复）。
    否则返回该 collection 下所有 file_name。"""
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            if (category or "").strip() == "project_case" and case_id is not None:
                cur.execute(
                    """SELECT DISTINCT file_name FROM knowledge_docs
                       WHERE collection = %s AND file_name IS NOT NULL AND file_name != ''
                       AND (category = %s OR category IS NULL OR category = '') AND case_id = %s""",
                    (collection, "project_case", case_id),
                )
            else:
                cur.execute(
                    "SELECT DISTINCT file_name FROM knowledge_docs WHERE collection = %s AND file_name IS NOT NULL AND file_name != ''",
                    (collection,),
                )
            return [r["file_name"] for r in cur.fetchall()]
    finally:
        conn.close()


def delete_knowledge_docs_by_file(
    collection: str,
    file_name: str,
    case_id: Optional[int] = None,
) -> int:
    """按文件名删除该知识库下对应文档的所有块（用于覆盖前清理）。
    当 case_id 有值时仅删除该 case_id 下的记录（项目案例覆盖时用）。"""
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            if case_id is not None:
                cur.execute(
                    "DELETE FROM knowledge_docs WHERE collection = %s AND file_name = %s AND case_id = %s",
                    (collection, file_name, case_id),
                )
            else:
                cur.execute(
                    "DELETE FROM knowledge_docs WHERE collection = %s AND file_name = %s",
                    (collection, file_name),
                )
            deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


def update_audit_report(report_id: int, report: Dict[str, Any]) -> None:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE audit_reports SET
                    report_json = %s,
                    total_points = %s,
                    high_count = %s,
                    medium_count = %s,
                    low_count = %s,
                    info_count = %s,
                    summary = %s
                WHERE id = %s
            """, (
                json.dumps(report, ensure_ascii=False),
                report.get("total_points", 0),
                report.get("high_count", 0),
                report.get("medium_count", 0),
                report.get("low_count", 0),
                report.get("info_count", 0),
                report.get("summary", ""),
                report_id,
            ))
        conn.commit()
    finally:
        conn.close()


def save_audit_correction(
    report_id: int,
    point_index: int,
    collection: str,
    file_name: str,
    original: Dict[str, Any],
    corrected: Dict[str, Any],
    fed_to_kb: bool = False,
) -> int:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO audit_corrections
                    (report_id, point_index, collection, file_name, original_json, corrected_json, fed_to_kb)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                report_id, point_index, collection or "", file_name or "",
                json.dumps(original, ensure_ascii=False),
                json.dumps(corrected, ensure_ascii=False),
                1 if fed_to_kb else 0,
            ))
            cid = cur.lastrowid
        conn.commit()
        return cid
    finally:
        conn.close()


def save_audit_checklist(
    collection: str,
    name: str,
    checklist: list,
    base_file: str = "",
    model_info: str = "",
    status: str = "draft",
    document_language: str = "",
) -> int:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO audit_checklists
                    (collection, name, checklist_json, total_points, base_file, model_info, status, document_language)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                collection or "",
                name,
                json.dumps(checklist, ensure_ascii=False),
                len(checklist),
                base_file,
                model_info,
                status,
                (document_language or "")[:32],
            ))
            cid = cur.lastrowid
        conn.commit()
        return cid
    finally:
        conn.close()


def get_audit_checklists(
    collection: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            sql = "SELECT * FROM audit_checklists WHERE 1=1"
            params = []
            if collection:
                sql += " AND collection = %s"
                params.append(collection)
            if status:
                sql += " AND status = %s"
                params.append(status)
            sql += " ORDER BY id DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            cur.execute(sql, params)
            rows = cur.fetchall()
        result = []
        for row in rows:
            r = dict(row)
            if r.get("checklist_json"):
                try:
                    r["checklist"] = json.loads(r["checklist_json"])
                except Exception:
                    r["checklist"] = []
            else:
                r["checklist"] = []
            result.append(r)
        return result
    finally:
        conn.close()


def get_audit_checklist_by_id(checklist_id: int) -> Optional[Dict[str, Any]]:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM audit_checklists WHERE id = %s", (checklist_id,))
            row = cur.fetchone()
        if not row:
            return None
        r = dict(row)
        r["checklist"] = json.loads(r.get("checklist_json") or "[]") if r.get("checklist_json") else []
        return r
    finally:
        conn.close()


def update_audit_checklist(checklist_id: int, checklist: list, name: str = None, status: str = None) -> None:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            sets = ["checklist_json = %s", "total_points = %s"]
            params = [json.dumps(checklist, ensure_ascii=False), len(checklist)]
            if name is not None:
                sets.append("name = %s")
                params.append(name)
            if status is not None:
                sets.append("status = %s")
                params.append(status)
            params.append(checklist_id)
            cur.execute(f"UPDATE audit_checklists SET {', '.join(sets)} WHERE id = %s", params)
        conn.commit()
    finally:
        conn.close()


def delete_audit_checklist(checklist_id: int) -> None:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM audit_checklists WHERE id = %s", (checklist_id,))
        conn.commit()
    finally:
        conn.close()


def get_corrections_for_collection(collection: str, limit: int = 200) -> list:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM audit_corrections
                WHERE collection = %s
                ORDER BY created_at DESC LIMIT %s
            """, (collection, limit))
            rows = cur.fetchall()
        result = []
        for row in rows:
            r = dict(row)
            for k in ("original_json", "corrected_json"):
                if r.get(k):
                    try:
                        r[k.replace("_json", "")] = json.loads(r[k])
                    except Exception:
                        r[k.replace("_json", "")] = {}
            result.append(r)
        return result
    finally:
        conn.close()


# 注册类别、注册组成：固定选项（不同国家可细分Ⅱa、Ⅱb 等，此处为统一列表）
REGISTRATION_TYPES = ["医疗器械一类", "医疗器械二类", "医疗器械二类Ⅱa", "医疗器械二类Ⅱb", "医疗器械三类"]
REGISTRATION_COMPONENTS = ["有源医疗器械", "软件组件", "独立软件"]


def get_dimension_options() -> Dict[str, Any]:
    """获取维度选项（注册国家、项目形态，页面可配置）"""
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM dimension_options WHERE id = 1")
            row = cur.fetchone()
        if not row:
            return {
                "registration_countries": ["中国", "美国", "欧盟"],
                "project_forms": ["Web", "APP", "PC"],
                "country_extra_keywords": {},
            }
        raw_kw = row.get("country_extra_keywords") or ""
        try:
            country_extra_keywords = json.loads(raw_kw) if raw_kw else {}
        except Exception:
            country_extra_keywords = {}
        if not isinstance(country_extra_keywords, dict):
            country_extra_keywords = {}
        return {
            "registration_countries": json.loads(row.get("registration_countries") or "[]") or ["中国", "美国", "欧盟"],
            "project_forms": json.loads(row.get("project_forms") or "[]") or ["Web", "APP", "PC"],
            "country_extra_keywords": country_extra_keywords,
        }
    except Exception:
        return {"registration_countries": ["中国", "美国", "欧盟"], "project_forms": ["Web", "APP", "PC"], "country_extra_keywords": {}}
    finally:
        conn.close()


def save_dimension_options(
    registration_countries: list = None,
    project_forms: list = None,
    country_extra_keywords: dict = None,
) -> None:
    """保存维度选项（仅更新传入的字段）。country_extra_keywords 为 {"CE": ["MDR"], "欧盟": ["MDR"]} 形式。"""
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM dimension_options WHERE id = 1")
            row = cur.fetchone()
            if not row:
                cur.execute("""
                    INSERT INTO dimension_options (id, registration_countries, project_forms, country_extra_keywords)
                    VALUES (1, %s, %s, %s)
                """, (
                    json.dumps(registration_countries or ["中国", "美国", "欧盟"], ensure_ascii=False),
                    json.dumps(project_forms or ["Web", "APP", "PC"], ensure_ascii=False),
                    json.dumps(country_extra_keywords if country_extra_keywords is not None else {}, ensure_ascii=False),
                ))
            else:
                updates = []
                params = []
                if registration_countries is not None:
                    updates.append("registration_countries = %s")
                    params.append(json.dumps(registration_countries, ensure_ascii=False))
                if project_forms is not None:
                    updates.append("project_forms = %s")
                    params.append(json.dumps(project_forms, ensure_ascii=False))
                if country_extra_keywords is not None:
                    updates.append("country_extra_keywords = %s")
                    params.append(json.dumps(country_extra_keywords, ensure_ascii=False))
                if updates:
                    params.append(1)
                    cur.execute("UPDATE dimension_options SET " + ", ".join(updates) + " WHERE id = %s", params)
        conn.commit()
    finally:
        conn.close()


def list_projects(collection: str) -> list:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM projects WHERE collection = %s ORDER BY id DESC",
                (collection,),
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_project(project_id: int) -> Optional[Dict[str, Any]]:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM projects WHERE id = %s", (project_id,))
            row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def create_project(
    collection: str,
    name: str,
    registration_country: str,
    registration_type: str,
    registration_component: str,
    project_form: str,
    scope_of_application: str = "",
    product_name: str = "",
    name_en: str = "",
    product_name_en: str = "",
    registration_country_en: str = "",
) -> int:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO projects (collection, name, registration_country, registration_type, registration_component, project_form, scope_of_application, product_name, name_en, product_name_en, registration_country_en)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (collection, name, registration_country, registration_type, registration_component, project_form, scope_of_application or "", product_name or "", name_en or "", product_name_en or "", registration_country_en or ""))
            pid = cur.lastrowid
        conn.commit()
        return pid
    finally:
        conn.close()


def update_project(
    project_id: int,
    name: str = None,
    registration_country: str = None,
    registration_type: str = None,
    registration_component: str = None,
    project_form: str = None,
    scope_of_application: str = None,
    product_name: str = None,
    name_en: str = None,
    product_name_en: str = None,
    registration_country_en: str = None,
) -> None:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            updates = []
            params = []
            for k, v in [
                ("name", name),
                ("registration_country", registration_country),
                ("registration_type", registration_type),
                ("registration_component", registration_component),
                ("project_form", project_form),
                ("scope_of_application", scope_of_application),
                ("product_name", product_name),
                ("name_en", name_en),
                ("product_name_en", product_name_en),
                ("registration_country_en", registration_country_en),
            ]:
                if v is not None:
                    updates.append(f"{k} = %s")
                    params.append(v)
            if updates:
                params.append(project_id)
                cur.execute("UPDATE projects SET " + ", ".join(updates) + " WHERE id = %s", params)
        conn.commit()
    finally:
        conn.close()


def delete_project(project_id: int) -> None:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM projects WHERE id = %s", (project_id,))
            cur.execute("DELETE FROM project_knowledge_docs WHERE project_id = %s", (project_id,))
        conn.commit()
    finally:
        conn.close()


def save_project_knowledge_docs(project_id: int, collection: str, file_name: str, chunks: list) -> int:
    init_db()
    if not chunks:
        return 0
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            rows = []
            for idx, doc in enumerate(chunks):
                content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                meta = doc.metadata if hasattr(doc, "metadata") else {}
                rows.append((project_id, collection, file_name, idx, content, json.dumps(meta, ensure_ascii=False)))
            cur.executemany("""
                INSERT INTO project_knowledge_docs (project_id, collection, file_name, chunk_index, content, metadata_json)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, rows)
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def get_project_knowledge_stats(project_id: int) -> Dict[str, Any]:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS total_chunks, COUNT(DISTINCT file_name) AS total_files FROM project_knowledge_docs WHERE project_id = %s",
                (project_id,),
            )
            return dict(cur.fetchone())
    finally:
        conn.close()


def clear_project_knowledge_docs(project_id: int) -> int:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM project_knowledge_docs WHERE project_id = %s", (project_id,))
            deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


def get_existing_project_file_names(project_id: int) -> list:
    """返回该项目下已存在的文件名列表（用于检测重复）"""
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT file_name FROM project_knowledge_docs WHERE project_id = %s AND file_name IS NOT NULL AND file_name != ''",
                (project_id,),
            )
            return [r["file_name"] for r in cur.fetchall()]
    finally:
        conn.close()


def delete_project_knowledge_docs_by_file(project_id: int, file_name: str) -> int:
    """按文件名删除该项目下对应文档的所有块（用于覆盖前清理）"""
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM project_knowledge_docs WHERE project_id = %s AND file_name = %s", (project_id, file_name))
            deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


def get_project_knowledge_text(project_id: int, max_chars: int = 15000) -> str:
    """从项目知识库中读取已入库的文本（用于训练后提取基本信息）。按 file_name, chunk_index 顺序拼接，总长度不超过 max_chars。"""
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT content FROM project_knowledge_docs WHERE project_id = %s ORDER BY file_name, chunk_index""",
                (project_id,),
            )
            rows = cur.fetchall()
        parts = []
        total = 0
        for row in rows:
            content = (row.get("content") or "").strip()
            if not content:
                continue
            if total + len(content) > max_chars:
                parts.append(content[: max_chars - total])
                break
            parts.append(content)
            total += len(content)
        return "\n\n".join(parts) if parts else ""
    finally:
        conn.close()


def update_project_basic_info(project_id: int, basic_info_text: str) -> None:
    """更新项目基本信息（从项目资料中提取后写入，审核时与待审文档一致性核对）。"""
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE projects SET basic_info_text = %s WHERE id = %s", (basic_info_text or "", project_id))
        conn.commit()
    finally:
        conn.close()


def update_project_system_functionality(
    project_id: int, system_functionality_text: str, system_functionality_source: str = ""
) -> None:
    """更新项目系统功能描述（来自安装包或 URL 识别，审核时与待审文档一致性核对）。source 可选：package / url。"""
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE projects SET system_functionality_text = %s, system_functionality_source = %s WHERE id = %s",
                (system_functionality_text or "", (system_functionality_source or "")[:64], project_id),
            )
        conn.commit()
    finally:
        conn.close()


def save_checkpoint_docs(collection: str, file_name: str, chunks: list) -> int:
    """审核点知识库块写入 MySQL（与 Chroma 双写，统计以 DB 为准）"""
    init_db()
    if not chunks:
        return 0
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            rows = []
            for idx, doc in enumerate(chunks):
                content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                meta = doc.metadata if hasattr(doc, "metadata") else {}
                rows.append((collection, file_name, idx, content, json.dumps(meta, ensure_ascii=False)))
            cur.executemany("""
                INSERT INTO checkpoint_docs (collection, file_name, chunk_index, content, metadata_json)
                VALUES (%s, %s, %s, %s, %s)
            """, rows)
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def get_checkpoint_stats(collection: str) -> Dict[str, Any]:
    """审核点知识库统计（以 DB 为准）"""
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS total_chunks, COUNT(DISTINCT file_name) AS total_files FROM checkpoint_docs WHERE collection = %s",
                (collection,),
            )
            row = cur.fetchone()
            return {"total_chunks": row["total_chunks"] or 0, "total_files": row["total_files"] or 0, "document_count": row["total_chunks"] or 0}
    finally:
        conn.close()


def clear_checkpoint_docs(collection: str) -> int:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM checkpoint_docs WHERE collection = %s", (collection,))
            deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


def get_existing_checkpoint_file_names(collection: str) -> list:
    """返回该审核点知识库下已存在的 file_name 列表（用于检测重名清单）"""
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT file_name FROM checkpoint_docs WHERE collection = %s AND file_name IS NOT NULL AND file_name != ''",
                (collection,),
            )
            return [r["file_name"] for r in cur.fetchall()]
    finally:
        conn.close()


def delete_checkpoint_docs_by_file(collection: str, file_name: str) -> int:
    """按 file_name 删除该审核点知识库下对应条目（用于覆盖前清理）"""
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM checkpoint_docs WHERE collection = %s AND file_name = %s", (collection, file_name))
            deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


# ─── project_cases（项目案例元数据）───
def create_project_case(
    collection: str,
    case_name: str,
    product_name: str = "",
    registration_country: str = "",
    registration_type: str = "",
    registration_component: str = "",
    project_form: str = "",
    scope_of_application: str = "",
    case_name_en: str = "",
    product_name_en: str = "",
    registration_country_en: str = "",
    document_language: str = "zh",
    project_key: str = "",
) -> int:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO project_cases
                    (collection, case_name, product_name, registration_country, registration_type,
                     registration_component, project_form, scope_of_application,
                     case_name_en, product_name_en, registration_country_en, document_language, project_key)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                collection, case_name, product_name or "", registration_country or "",
                registration_type or "", registration_component or "",
                project_form or "", scope_of_application or "",
                case_name_en or "", product_name_en or "", registration_country_en or "",
                (document_language if document_language in ("", "zh", "en", "both") else "zh")[:32],
                (project_key or "")[:256],
            ))
            cid = cur.lastrowid
        conn.commit()
        return cid
    finally:
        conn.close()


def update_project_case(
    case_id: int,
    case_name: str = None,
    case_name_en: str = None,
    product_name: str = None,
    product_name_en: str = None,
    registration_country: str = None,
    registration_country_en: str = None,
    registration_type: str = None,
    registration_component: str = None,
    project_form: str = None,
    scope_of_application: str = None,
    document_language: str = None,
    project_key: str = None,
) -> None:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            updates = []
            params = []
            for k, v in [
                ("case_name", case_name),
                ("case_name_en", case_name_en),
                ("product_name", product_name),
                ("product_name_en", product_name_en),
                ("registration_country", registration_country),
                ("registration_country_en", registration_country_en),
                ("registration_type", registration_type),
                ("registration_component", registration_component),
                ("project_form", project_form),
                ("scope_of_application", scope_of_application),
                ("document_language", document_language),
                ("project_key", project_key if project_key is None else (project_key[:256] if project_key else "")),
            ]:
                if v is not None:
                    updates.append(f"{k} = %s")
                    params.append(v)
            if updates:
                params.append(case_id)
                cur.execute("UPDATE project_cases SET " + ", ".join(updates) + " WHERE id = %s", params)
        conn.commit()
    finally:
        conn.close()


def list_project_cases(collection: str) -> list:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM project_cases WHERE collection = %s ORDER BY project_key, id DESC",
                (collection,),
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def list_project_case_project_keys(collection: str) -> list:
    """返回该知识库下已使用的关联项目标识列表（非空、去重），用于新建案例时选择「关联项目」。"""
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT project_key FROM project_cases WHERE collection = %s AND project_key IS NOT NULL AND project_key != '' ORDER BY project_key",
                (collection,),
            )
            return [row["project_key"] for row in cur.fetchall()]
    finally:
        conn.close()


def get_project_case(case_id: int) -> Optional[Dict[str, Any]]:
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM project_cases WHERE id = %s", (case_id,))
            row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_project_case_file_names(collection: str, case_id: int) -> list:
    """返回该项目案例在知识库中已入库的文件名列表（knowledge_docs 中 case_id 且 category=project_case）。"""
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT DISTINCT file_name FROM knowledge_docs
                   WHERE collection = %s AND (category = %s OR category IS NULL OR category = '') AND case_id = %s
                   AND file_name IS NOT NULL AND file_name != '' ORDER BY file_name""",
                (collection, "project_case", case_id),
            )
            return [r["file_name"] for r in cur.fetchall()]
    finally:
        conn.close()


def delete_project_case(case_id: int) -> bool:
    """删除项目案例记录。调用前需确保该案例下无已入库文件（否则不应删除）。"""
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM project_cases WHERE id = %s", (case_id,))
            conn.commit()
            return cur.rowcount > 0
    finally:
        conn.close()


def update_knowledge_docs_case_id(collection: str, file_name: str, case_id: int) -> None:
    """将指定知识库文件的所有块关联到 case_id。"""
    init_db()
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE knowledge_docs SET case_id = %s WHERE collection = %s AND file_name = %s",
                (case_id, collection, file_name),
            )
        conn.commit()
    finally:
        conn.close()
