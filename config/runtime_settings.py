"""
运行时配置与数据库 `runtime_settings_json` 同步：迁移时仅需 .env 中 MySQL 可连库，其余从库载入。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import TypeAdapter

from config import settings


def list_persistable_setting_keys() -> List[str]:
    """可持久化到 JSON 的 Settings 字段（不含 model_config）。"""
    return [k for k in settings.model_fields if k != "model_config"]


def serialize_settings_to_flat_dict() -> Dict[str, Any]:
    """从当前内存中的 settings 导出为可 JSON 序列化的字典。"""
    out: Dict[str, Any] = {}
    for key in list_persistable_setting_keys():
        val = getattr(settings, key, None)
        if hasattr(val, "isoformat"):
            val = val.isoformat()
        out[key] = val
    return out


def apply_runtime_config_dict(data: Optional[Dict[str, Any]]) -> int:
    """
    将字典写回全局 settings（按字段类型校验）。返回成功写入的键数量。
    """
    if not data:
        return 0
    n = 0
    for key, field in settings.model_fields.items():
        if key == "model_config" or key not in data:
            continue
        raw = data[key]
        try:
            ta = TypeAdapter(field.annotation)
            val = ta.validate_python(raw)
            setattr(settings, key, val)
            n += 1
        except Exception:
            continue
    return n


def sync_cursor_overrides_from_settings() -> None:
    """载入配置后：让 cursor_overrides 回退为「使用 settings」，避免侧栏曾写入的旧覆盖残留。"""
    try:
        from config.cursor_overrides import _cursor_overrides

        _cursor_overrides["verify_ssl"] = None
        _cursor_overrides["trust_env"] = None
    except Exception:
        pass


def export_dotenv_lines() -> str:
    """生成可粘贴到 .env 的文本（字段名大写，便于新机器最小启动）。"""
    lines: List[str] = []
    lines.append("# 由系统配置页导出；新机器至少保证 MYSQL_* 能连上数据库，其余可由库内 runtime_settings_json 覆盖")
    for key in sorted(list_persistable_setting_keys()):
        val = getattr(settings, key, None)
        if val is None:
            continue
        if isinstance(val, bool):
            s = "true" if val else "false"
        else:
            s = str(val)
            if "\n" in s or "#" in s:
                s = json.dumps(s, ensure_ascii=False)
        env = key.upper()
        lines.append(f"{env}={s}")
    return "\n".join(lines) + "\n"


def merge_runtime_json_into_row(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """若行内存在 runtime_settings_json，解析为 dict 供 UI 使用（不修改全局 settings）。"""
    if not row:
        return row
    blob = row.get("runtime_settings_json")
    if not blob:
        return row
    try:
        parsed = json.loads(blob) if isinstance(blob, str) else blob
        if isinstance(parsed, dict):
            row = dict(row)
            row["_runtime_settings_parsed"] = parsed
    except Exception:
        pass
    return row
