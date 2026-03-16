"""所有 AI 服务通用的 HTTP 运行时覆盖（SSL 校验、代理）。侧栏勾选会写回此处与 settings.llm_*。"""

_cursor_overrides: dict = {"verify_ssl": None, "trust_env": None}


def get_llm_verify_ssl() -> bool:
    """是否校验 SSL，供所有 AI 服务（OpenAI/DeepSeek/零一/Cursor/Ollama 等）使用。"""
    v = _cursor_overrides.get("verify_ssl")
    if v is None:
        try:
            from config import settings
            return getattr(settings, "llm_verify_ssl", getattr(settings, "cursor_verify_ssl", True))
        except Exception:
            return True
    return bool(v)


def get_llm_trust_env() -> bool:
    """是否使用系统代理，供所有 AI 服务使用。False 表示直连、不使用代理。"""
    v = _cursor_overrides.get("trust_env")
    if v is None:
        try:
            from config import settings
            return getattr(settings, "llm_trust_env", getattr(settings, "cursor_trust_env", True))
        except Exception:
            return True
    return bool(v)


def get_cursor_verify_ssl() -> bool:
    """兼容旧名，等同于 get_llm_verify_ssl()。"""
    return get_llm_verify_ssl()


def get_cursor_trust_env() -> bool:
    """兼容旧名，等同于 get_llm_trust_env()。"""
    return get_llm_trust_env()
