"""Cursor 请求的运行时覆盖（SSL 校验、代理），避免直接修改 Pydantic Settings"""

_cursor_overrides: dict = {"verify_ssl": None, "trust_env": None}


def get_cursor_verify_ssl() -> bool:
    v = _cursor_overrides.get("verify_ssl")
    if v is None:
        try:
            from config import settings
            return getattr(settings, "cursor_verify_ssl", True)
        except Exception:
            return True
    return bool(v)


def get_cursor_trust_env() -> bool:
    v = _cursor_overrides.get("trust_env")
    if v is None:
        try:
            from config import settings
            return getattr(settings, "cursor_trust_env", True)
        except Exception:
            return True
    return bool(v)
