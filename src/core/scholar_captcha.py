"""Google Scholar 人机验证会话：同一 llm_http_proxy 出口代理验证页，供用户在弹窗中完成。"""

from __future__ import annotations

import re
import threading
import time
import uuid
from typing import Any, Optional
from urllib.parse import parse_qsl, quote_plus, unquote, urljoin, urlparse

import httpx

_SESSION_TTL_SECONDS = 30 * 60
_LOCK = threading.Lock()
_SESSIONS: dict[str, dict[str, Any]] = {}

_ALLOWED_HOST_SUFFIXES = (
    "google.com",
    "google.com.hk",
    "gstatic.com",
    "googleapis.com",
    "googleusercontent.com",
)


def _allowed_url(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    if not host:
        return False
    return any(host == s or host.endswith("." + s) for s in _ALLOWED_HOST_SUFFIXES)


def _purge_expired_unlocked() -> None:
    now = time.time()
    dead = [k for k, v in _SESSIONS.items() if now - float(v.get("created_at") or 0) > _SESSION_TTL_SECONDS]
    for k in dead:
        _SESSIONS.pop(k, None)


def create_scholar_captcha_session(
    *,
    captcha_url: str,
    cookies: dict[str, str],
    html: str = "",
    search_url: str = "",
) -> str:
    sid = uuid.uuid4().hex
    with _LOCK:
        _purge_expired_unlocked()
        _SESSIONS[sid] = {
            "created_at": time.time(),
            "captcha_url": (captcha_url or "").strip(),
            "search_url": (search_url or "").strip(),
            "cookies": dict(cookies or {}),
            "html": html or "",
            "last_url": (captcha_url or "").strip(),
        }
    return sid


def get_scholar_captcha_session(session_id: str) -> Optional[dict[str, Any]]:
    sid = (session_id or "").strip()
    if not sid:
        return None
    with _LOCK:
        _purge_expired_unlocked()
        row = _SESSIONS.get(sid)
        if not row:
            return None
        return dict(row)


def update_scholar_captcha_cookies(session_id: str, cookies: dict[str, str], *, last_url: str = "") -> None:
    sid = (session_id or "").strip()
    if not sid:
        return
    with _LOCK:
        row = _SESSIONS.get(sid)
        if not row:
            return
        merged = dict(row.get("cookies") or {})
        merged.update({str(k): str(v) for k, v in (cookies or {}).items()})
        row["cookies"] = merged
        if last_url:
            row["last_url"] = last_url
        row["created_at"] = time.time()


def _build_client(*, timeout_seconds: float = 30) -> httpx.Client:
    from config.cursor_overrides import get_llm_http_proxy, get_llm_verify_ssl

    proxy = (get_llm_http_proxy() or "").strip()
    verify = bool(get_llm_verify_ssl())
    kwargs: dict[str, Any] = {
        "trust_env": False,
        "verify": verify,
        "timeout": httpx.Timeout(timeout_seconds, connect=min(15.0, timeout_seconds)),
        "follow_redirects": True,
        "http2": False,
    }
    if proxy:
        kwargs["proxy"] = proxy
    return httpx.Client(**kwargs)


def _cookie_jar(cookies: dict[str, str]) -> httpx.Cookies:
    jar = httpx.Cookies()
    for k, v in (cookies or {}).items():
        try:
            jar.set(str(k), str(v), domain=".google.com")
        except Exception:
            try:
                jar.set(str(k), str(v))
            except Exception:
                pass
    return jar


def _rewrite_html(content: str, *, rewrite_base: str, current_url: str) -> str:
    base = (rewrite_base or "").rstrip("/")
    if not base:
        return content

    def _abs(u: str) -> str:
        return urljoin(current_url, u)

    def _proxied(abs_url: str) -> str:
        return f"{base}/nav?u={quote_plus(abs_url)}"

    def repl(match: re.Match[str]) -> str:
        attr, quote, raw = match.group(1), match.group(2), match.group(3)
        if raw.startswith(("javascript:", "data:", "mailto:", "#")):
            return match.group(0)
        abs_url = _abs(raw)
        if not _allowed_url(abs_url):
            return match.group(0)
        return f"{attr}={quote}{_proxied(abs_url)}{quote}"

    out = re.sub(
        r"""\b(href|src|action)=([\"'])([^\"']+)\2""",
        repl,
        content,
        flags=re.I,
    )
    out = re.sub(
        r"""(?is)<meta[^>]+http-equiv=["']?content-security-policy["']?[^>]*>""",
        "",
        out,
    )
    # 提示条：方便用户知道这是代理验证页
    banner = (
        "<div style='position:sticky;top:0;z-index:9999;background:#fff3cd;"
        "border-bottom:1px solid #ffecb5;padding:8px 12px;font:14px/1.4 sans-serif;'>"
        "请在本页完成 Google 人机验证。完成后请回到原窗口点击「验证完成，继续检索」。"
        "</div>"
    )
    if re.search(r"(?is)<body[^>]*>", out):
        out = re.sub(r"(?is)(<body[^>]*>)", r"\1" + banner, out, count=1)
    else:
        out = banner + out
    return out


def proxy_scholar_captcha(
    session_id: str,
    *,
    method: str = "GET",
    target_url: str = "",
    form_data: Optional[dict[str, str]] = None,
    rewrite_base: str = "",
) -> tuple[int, dict[str, str], bytes, str]:
    """代理验证页请求。返回 (status, headers, body, content_type)。"""
    sess = get_scholar_captcha_session(session_id)
    if not sess:
        return 404, {"content-type": "text/plain; charset=utf-8"}, b"captcha session expired", "text/plain"

    url = (target_url or sess.get("captcha_url") or sess.get("last_url") or "").strip()
    if not url:
        return 400, {"content-type": "text/plain; charset=utf-8"}, b"missing captcha url", "text/plain"
    if not _allowed_url(url):
        return 400, {"content-type": "text/plain; charset=utf-8"}, b"url not allowed", "text/plain"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    # 首次若有缓存 HTML 且未指定其它 URL，直接改写返回，减少重复触发
    if method.upper() == "GET" and not target_url and (sess.get("html") or "").strip():
        html = _rewrite_html(str(sess.get("html") or ""), rewrite_base=rewrite_base, current_url=url)
        return 200, {"content-type": "text/html; charset=utf-8"}, html.encode("utf-8", errors="ignore"), "text/html"

    try:
        with _build_client() as client:
            client.cookies = _cookie_jar(sess.get("cookies") or {})
            if method.upper() == "POST":
                resp = client.post(url, data=form_data or {}, headers=headers)
            else:
                resp = client.get(url, headers=headers)
            update_scholar_captcha_cookies(
                session_id,
                {k: v for k, v in resp.cookies.items()},
                last_url=str(resp.url),
            )
            ctype = (resp.headers.get("content-type") or "application/octet-stream").split(";")[0].strip()
            body = resp.content or b""
            if "text/html" in ctype or body[:200].lstrip().lower().startswith((b"<!doctype", b"<html")):
                text = body.decode(resp.encoding or "utf-8", errors="ignore")
                text = _rewrite_html(text, rewrite_base=rewrite_base, current_url=str(resp.url))
                body = text.encode("utf-8", errors="ignore")
                ctype = "text/html"
            out_headers = {
                "content-type": f"{ctype}; charset=utf-8" if ctype.startswith("text/") else ctype,
                # 允许嵌入到 aiword 弹窗 iframe
                "cache-control": "no-store",
            }
            return int(resp.status_code), out_headers, body, ctype
    except Exception as exc:
        msg = f"proxy captcha failed: {exc}".encode("utf-8", errors="ignore")
        return 502, {"content-type": "text/plain; charset=utf-8"}, msg, "text/plain"


def decode_nav_url(raw: str) -> str:
    return unquote((raw or "").strip())


def parse_form_body(body: bytes, content_type: str) -> dict[str, str]:
    ctype = (content_type or "").lower()
    if "application/x-www-form-urlencoded" in ctype:
        return {k: v for k, v in parse_qsl(body.decode("utf-8", errors="ignore"), keep_blank_values=True)}
    return {}
