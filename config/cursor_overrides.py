"""所有 AI 服务通用的 HTTP 运行时覆盖（SSL 校验、代理）。侧栏勾选会写回此处与 settings.llm_*。"""

import os
from typing import Any, Optional, Union

import httpx

from config.http_proxy_policy import foreign_proxy_mount_patterns, proxy_for_url

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


def get_llm_http_proxy() -> Optional[str]:
    """显式 HTTP(S) 代理地址。优先 settings.llm_http_proxy，其次 .env 的 HTTPS_PROXY/HTTP_PROXY。"""
    try:
        from config import settings

        explicit = (getattr(settings, "llm_http_proxy", None) or "").strip()
        if explicit:
            return explicit
    except Exception:
        pass
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    return None


def _base_httpx_kwargs() -> dict[str, Any]:
    return {
        "http2": False,
        "limits": httpx.Limits(max_keepalive_connections=5, max_connections=10),
        "verify": get_llm_verify_ssl(),
    }


def llm_httpx_client_kwargs(*, for_url: Optional[str] = None) -> dict[str, Any]:
    """供 httpx.Client 或 Ollama client_kwargs 使用。for_url 已知时仅在国外 AI 域名上挂 proxy。"""
    kwargs = _base_httpx_kwargs()
    proxy_base = get_llm_http_proxy()
    if for_url and proxy_base:
        explicit = proxy_for_url(for_url, proxy_base=proxy_base)
        if explicit:
            kwargs["proxy"] = explicit
            kwargs["trust_env"] = False
            return kwargs
    kwargs["trust_env"] = get_llm_trust_env()
    return kwargs


def _build_selective_proxy_mounts(proxy_url: str, verify: bool) -> dict[str, httpx.HTTPTransport]:
    """默认直连；仅国外 AI 域名 mount 走代理。"""
    direct = httpx.HTTPTransport(verify=verify)
    proxied = httpx.HTTPTransport(proxy=proxy_url, verify=verify)
    mounts: dict[str, httpx.HTTPTransport] = {"all://": direct}
    for pattern in foreign_proxy_mount_patterns():
        mounts[pattern] = proxied
    return mounts


def build_llm_httpx_client(
    *,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    for_url: Optional[str] = None,
) -> httpx.Client:
    """创建 httpx 客户端：国内/内网直连，仅国外 AI 域名经代理。"""
    proxy_base = get_llm_http_proxy()
    verify = get_llm_verify_ssl()

    if for_url and proxy_base:
        kwargs = llm_httpx_client_kwargs(for_url=for_url)
        if timeout is not None:
            kwargs["timeout"] = timeout
        return httpx.Client(**kwargs)

    if proxy_base:
        client_kwargs: dict[str, Any] = {
            **_base_httpx_kwargs(),
            "mounts": _build_selective_proxy_mounts(proxy_base, verify),
            "trust_env": False,
        }
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        return httpx.Client(**client_kwargs)

    kwargs = llm_httpx_client_kwargs()
    if timeout is not None:
        kwargs["timeout"] = timeout
    return httpx.Client(**kwargs)
