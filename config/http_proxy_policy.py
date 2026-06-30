"""按目标主机决定是否走 HTTP 代理：国内/内网直连，仅国外 AI 域名走代理。"""

from __future__ import annotations

import ipaddress
import os
from typing import Iterable, Optional, Set
from urllib.parse import urlparse

# 匹配主机名后缀（含子域）；优先于国外列表
_DOMESTIC_HOST_SUFFIXES: tuple[str, ...] = (
    ".aliyuncs.com",
    ".deepseek.com",
    ".lingyiwanwu.com",
    ".baidubce.com",
    ".baidu.com",
    ".qianfan.baidu.com",
    ".volces.com",
    ".bigmodel.cn",
    ".moonshot.cn",
    ".zhipuai.cn",
)

# 仅这些后缀（及子域）走代理；其余默认直连
_FOREIGN_AI_HOST_SUFFIXES: tuple[str, ...] = (
    ".cursor.com",
    ".openai.com",
    ".anthropic.com",
    ".openai.azure.com",
    ".googleapis.com",
    ".azure.com",
)

_LOCAL_HOSTNAMES: frozenset[str] = frozenset({"localhost", "localhost.localdomain"})


def _host_suffixes_from_settings(attr: str) -> tuple[str, ...]:
    try:
        from config import settings

        raw = (getattr(settings, attr, None) or "").strip()
    except Exception:
        raw = ""
    if not raw:
        return ()
    out: list[str] = []
    for part in raw.replace(";", ",").split(","):
        s = part.strip().lower()
        if not s:
            continue
        if not s.startswith(".") and "." in s and not s.replace(".", "").isdigit():
            s = f".{s}" if not s.startswith("*") else s
        out.append(s)
    return tuple(out)


def domestic_host_suffixes() -> tuple[str, ...]:
    return _DOMESTIC_HOST_SUFFIXES + _host_suffixes_from_settings("llm_proxy_domestic_suffixes")


def foreign_ai_host_suffixes() -> tuple[str, ...]:
    return _FOREIGN_AI_HOST_SUFFIXES + _host_suffixes_from_settings("llm_proxy_foreign_suffixes")


def _normalize_hostname(host: Optional[str]) -> str:
    return (host or "").strip().lower().rstrip(".")


def _is_private_or_local_host(host: str) -> bool:
    if not host:
        return True
    if host in _LOCAL_HOSTNAMES:
        return True
    if host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return bool(
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
        )
    except ValueError:
        pass
    return False


def _host_matches_suffix(host: str, suffixes: Iterable[str]) -> bool:
    for suf in suffixes:
        s = suf.strip().lower()
        if not s:
            continue
        if s.startswith("*."):
            s = s[1:]
        if not s.startswith("."):
            if host == s:
                return True
            s = f".{s}"
        if host == s[1:] or host.endswith(s):
            return True
    return False


def should_use_foreign_proxy(host: Optional[str]) -> bool:
    """True 表示应经 HTTP_PROXY / llm_http_proxy 访问。"""
    h = _normalize_hostname(host)
    if not h:
        return False
    if _is_private_or_local_host(h):
        return False
    if _host_matches_suffix(h, domestic_host_suffixes()):
        return False
    return _host_matches_suffix(h, foreign_ai_host_suffixes())


def proxy_for_url(url: str, *, proxy_base: Optional[str] = None) -> Optional[str]:
    """若 url 属于国外 AI 域名则返回代理地址，否则 None（直连）。"""
    base = (proxy_base or "").strip()
    if not base:
        try:
            from config.cursor_overrides import get_llm_http_proxy

            base = (get_llm_http_proxy() or "").strip()
        except Exception:
            base = ""
    if not base:
        return None
    try:
        host = urlparse(url).hostname
    except Exception:
        return None
    if should_use_foreign_proxy(host):
        return base
    return None


def foreign_proxy_mount_patterns() -> list[str]:
    """httpx mounts 键：仅国外 AI 域名挂代理 transport。"""
    patterns: Set[str] = set()
    for suf in foreign_ai_host_suffixes():
        s = suf.strip().lower()
        if not s:
            continue
        if s.startswith("*."):
            patterns.add(f"all://{s}")
            continue
        if s.startswith("."):
            patterns.add(f"all://*{s}")
            patterns.add(f"all://{s[1:]}")
        else:
            patterns.add(f"all://{s}")
            patterns.add(f"all://*.{s}")
    return sorted(patterns)


def bootstrap_no_proxy_env() -> None:
    """为 DashScope 等 SDK 设置 NO_PROXY，避免读系统 HTTP_PROXY 时误走代理。"""
    entries: Set[str] = {
        "localhost",
        "127.0.0.1",
        "::1",
        ".local",
    }
    for suf in domestic_host_suffixes():
        if suf.startswith("."):
            entries.add(suf)
    try:
        from config import settings

        for raw in (
            getattr(settings, "ollama_base_url", "") or "",
            getattr(settings, "chroma_server_host", "") or "",
            getattr(settings, "mysql_host", "") or "",
        ):
            h = _normalize_hostname(urlparse(raw if "://" in raw else f"//{raw}").hostname or raw.split(":")[0])
            if h:
                entries.add(h)
    except Exception:
        pass
    merged: list[str] = []
    for key in ("NO_PROXY", "no_proxy"):
        cur = os.environ.get(key, "")
        if cur:
            merged.extend(x.strip() for x in cur.split(",") if x.strip())
    merged.extend(sorted(entries))
    # 去重保序
    seen: Set[str] = set()
    final: list[str] = []
    for x in merged:
        xl = x.lower()
        if xl not in seen:
            seen.add(xl)
            final.append(x)
    val = ",".join(final)
    os.environ["NO_PROXY"] = val
    os.environ["no_proxy"] = val
