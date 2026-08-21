"""
调用 Cursor Cloud Agents API：发起任务、轮询完成、取回助手回复文本。
支持请求级凭据（``ClientLlmConfig`` / aiword Header）与 ``settings`` 合并。
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from typing import Any, Callable, Optional, TypeVar

import httpx

from config import settings

_log = logging.getLogger(__name__)

_T = TypeVar("_T")

# 本进程内：HTTP 代理对 Cursor 出现 SSL EOF 后，后续请求一律直连（避免每次先打坏代理再 429）
_skip_broken_http_proxy = False
_proxy_dead_logged = False
_launch_lock = threading.Lock()
_last_launch_monotonic = 0.0
_MIN_LAUNCH_INTERVAL_SEC = 2.0
_MAX_429_ATTEMPTS = 4

# 代理/弱网下常见的可重试网络错误
_TRANSIENT_NET_EXC = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.TimeoutException,
    OSError,
)

_CURSOR_HTTP_MAX_ATTEMPTS = 4
_CURSOR_HTTP_RETRY_BASE_SEC = 2.0


def _auth_header(api_key: str) -> str:
    raw = f"{api_key}:"
    return "Basic " + base64.b64encode(raw.encode()).decode()


def _get_headers(api_key: str) -> dict:
    return {
        "Authorization": _auth_header(api_key),
        "Content-Type": "application/json",
    }


def _base_url(base: str) -> str:
    return (base or "https://api.cursor.com").rstrip("/")


def _http_timeout(timeout: float) -> httpx.Timeout:
    if isinstance(timeout, httpx.Timeout):
        return timeout
    return httpx.Timeout(timeout, connect=min(45.0, max(15.0, timeout * 0.25)))


def _http_client(timeout: float = 60, *, force_direct: bool = False, for_url: str = "") -> httpx.Client:
    """统一创建 httpx 客户端。支持：关闭 SSL 校验、显式/环境变量代理。"""
    t = _http_timeout(timeout)
    use_direct = force_direct or _skip_broken_http_proxy
    if use_direct:
        try:
            from config.cursor_overrides import get_llm_verify_ssl

            verify = get_llm_verify_ssl()
        except Exception:
            verify = True
        return httpx.Client(timeout=t, verify=verify, trust_env=False, http2=False)
    try:
        from config.cursor_overrides import build_llm_httpx_client

        return build_llm_httpx_client(timeout=t, for_url=(for_url or None))
    except Exception:
        return httpx.Client(timeout=timeout, verify=True, trust_env=True, http2=False)


def _is_broken_local_proxy(exc: BaseException) -> bool:
    """本地 HTTP_PROXY 实际是 SOCKS/空端口/非 CONNECT 时的典型失败。"""
    el = str(exc).lower()
    errno = getattr(exc, "errno", None)
    cause = getattr(exc, "__cause__", None)
    if errno is None and cause is not None:
        errno = getattr(cause, "errno", None)
    if errno == 2:
        return True
    return any(
        x in el
        for x in (
            "eof occurred in violation of protocol",
            "ssleof",
            "wrong version number",
            "unable to connect to proxy",
            "no such file or directory",
        )
    )


def _is_direct_blocked(exc: BaseException) -> bool:
    """直连被墙/重置时，才值得改走 HTTP 代理。"""
    if _is_broken_local_proxy(exc):
        return False
    el = str(exc).lower()
    return any(
        x in el
        for x in (
            "10054",
            "10060",
            "10061",
            "timed out",
            "timeout",
            "connection reset",
            "connection aborted",
            "network is unreachable",
        )
    )


def _is_transient_network_error(exc: BaseException) -> bool:
    if _is_broken_local_proxy(exc):
        return False
    el = str(exc).lower()
    return any(
        x in el
        for x in (
            "10054",
            "connection",
            "reset",
            "eof",
            "ssl",
            "timeout",
            "broken pipe",
            "handshake",
            "protocol",
        )
    )


def _cursor_connect_runtime_error(base: str, exc: BaseException) -> RuntimeError:
    extra = ""
    if _is_broken_local_proxy(exc):
        extra = (
            " 当前失败像是本地 HTTP_PROXY 不可用（常见把 Clash socks-port 填成 HTTP 代理，"
            "或 7897 不是 mixed-port）。本机若能直连 api.cursor.com，请清空 .env.txt 的 HTTP_PROXY。"
        )
    return RuntimeError(
        f"无法连接 Cursor API（{_base_url(base)}）：{exc}。"
        f"{extra}"
        " 也可改用通义/DeepSeek。"
    )


def _with_http_retry(
    op: Callable[[], _T],
    *,
    context: str = "cursor_http",
    max_attempts: int = _CURSOR_HTTP_MAX_ATTEMPTS,
) -> _T:
    last: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return op()
        except _TRANSIENT_NET_EXC as e:
            last = e
            if _is_broken_local_proxy(e):
                raise
            if attempt < max_attempts and _is_transient_network_error(e):
                time.sleep(_CURSOR_HTTP_RETRY_BASE_SEC * attempt)
                continue
            raise
    if last is not None:
        raise last
    raise RuntimeError(f"{context}: 请求失败")


def _mark_http_proxy_dead(exc: BaseException) -> None:
    global _skip_broken_http_proxy, _proxy_dead_logged
    _skip_broken_http_proxy = True
    if not _proxy_dead_logged:
        _proxy_dead_logged = True
        _log.warning(
            "Cursor API 经 HTTP 代理失败（%s），本进程改为直连。"
            "请确认 Clash mixed-port，不要把 socks-port 写进 HTTP_PROXY。",
            exc,
        )


def _retry_after_seconds(resp: httpx.Response, attempt: int) -> float:
    raw = (resp.headers.get("Retry-After") or resp.headers.get("retry-after") or "").strip()
    if raw:
        try:
            return min(120.0, max(3.0, float(raw)))
        except ValueError:
            pass
    return min(60.0, 8.0 * (2 ** max(0, attempt - 1)))


def _http_request(
    method: str,
    url: str,
    *,
    headers: dict,
    json: Optional[dict] = None,
    timeout: float = 60,
    context: str = "",
) -> httpx.Response:
    def _once(*, force_direct: bool = False) -> httpx.Response:
        with _http_client(timeout=timeout, force_direct=force_direct, for_url=url) as client:
            if method.upper() == "POST":
                return client.post(url, json=json, headers=headers)
            return client.get(url, headers=headers)

    def _do() -> httpx.Response:
        # 本机直连 api.cursor.com 通常可用；先直连，避免 .env HTTP_PROXY 指向 SOCKS 口导致 Errno 2 / SSL EOF。
        try:
            return _once(force_direct=True)
        except _TRANSIENT_NET_EXC as direct_exc:
            if _skip_broken_http_proxy or not _is_direct_blocked(direct_exc):
                raise
            try:
                return _once(force_direct=False)
            except _TRANSIENT_NET_EXC as proxy_exc:
                if _is_broken_local_proxy(proxy_exc):
                    _mark_http_proxy_dead(proxy_exc)
                raise direct_exc from proxy_exc

    last_429: Optional[httpx.Response] = None
    for attempt in range(1, _MAX_429_ATTEMPTS + 1):
        try:
            r = _with_http_retry(_do, context=context or url)
        except _TRANSIENT_NET_EXC as e:
            raise _cursor_connect_runtime_error(url, e) from e
        if r.status_code != 429:
            return r
        last_429 = r
        if attempt >= _MAX_429_ATTEMPTS:
            break
        wait = _retry_after_seconds(r, attempt)
        _log.warning(
            "Cursor API HTTP 429（%s），%s 秒后重试 %s/%s",
            context or method,
            int(wait),
            attempt,
            _MAX_429_ATTEMPTS,
        )
        time.sleep(wait)
    return last_429 if last_429 is not None else r


def _raise_for_status_with_body(r: httpx.Response, context: str = "") -> None:
    """4xx/5xx 时抛出包含响应体的异常，便于排查 404 等配置错误。"""
    if r.is_success:
        return
    try:
        body = r.json()
        if isinstance(body, dict):
            msg = body.get("error_msg") or body.get("message") or body.get("error") or str(body)
            if body.get("event_id"):
                msg = f"{msg} (event_id: {body.get('event_id')})"
        else:
            msg = r.text or f"HTTP {r.status_code}"
    except Exception:
        msg = r.text or f"HTTP {r.status_code}"
    hint = ""
    if r.status_code == 429:
        hint = (
            " Cursor Cloud Agents 触发限流。请等待几分钟后再提交；"
            " 分段审核/多文件会并发创建多个 Agent，短时间重复点提交会打满配额。"
            " 也可改用通义/DeepSeek。"
        )
    if r.status_code == 404:
        hint = " 请检查：1) Cursor API 基地址是否为 https://api.cursor.com；2) GitHub 仓库地址与分支/ref 是否正确且可访问；3) API Key 是否有效（Cursor Dashboard → Integrations）。"
    if r.status_code == 400 and "region" in (msg or "").lower():
        hint = (
            " 当前 Cursor 账号/请求来源地区不在 Cloud Agents 支持范围内（与网络是否连通无关）。"
            " 见 https://cursor.com/docs/account/regions ；可换支持地区的网络/账号，或改用通义/DeepSeek/OpenAI 中转等提供方。"
        )
    raise RuntimeError(f"Error code: {r.status_code} - {msg}{hint}".strip())


def launch_agent(prompt_text: str, *, client_llm: Optional[Any] = None) -> str:
    from src.core.llm_factory import ClientLlmConfig, merged_cursor_launch_params

    cl = client_llm if isinstance(client_llm, ClientLlmConfig) else None
    p = merged_cursor_launch_params(cl)
    if not p["api_key"] or not p["repository"]:
        raise RuntimeError(
            "Cursor 模式下请配置 API Key 与 GitHub 仓库地址（请求头 X-Client-Llm-Api-Key / "
            "X-Client-Cursor-Repository，或 aicheckword 系统设置中的 cursor_*）"
        )
    url = f"{_base_url(p['base_url'])}/v0/agents"
    payload = {
        "prompt": {"text": prompt_text},
        "source": {
            "repository": p["repository"].strip(),
            "ref": (p["ref"] or "main").strip(),
        },
        "target": {"autoCreatePr": False},
    }
    global _last_launch_monotonic
    with _launch_lock:
        gap = time.monotonic() - _last_launch_monotonic
        if _last_launch_monotonic > 0 and gap < _MIN_LAUNCH_INTERVAL_SEC:
            time.sleep(_MIN_LAUNCH_INTERVAL_SEC - gap)
        r = _http_request(
            "POST",
            url,
            json=payload,
            headers=_get_headers(p["api_key"]),
            timeout=90,
            context="launch_agent",
        )
        _last_launch_monotonic = time.monotonic()
    _raise_for_status_with_body(r, "launch_agent")
    data = r.json()
    return data["id"]


# 轮询状态/拉取对话时单次请求超时（秒），避免多文档等长任务时 read timeout
_POLL_REQUEST_TIMEOUT = 120


def get_agent_status(agent_id: str, *, client_llm: Optional[Any] = None) -> dict:
    from src.core.llm_factory import ClientLlmConfig, merged_cursor_launch_params

    cl = client_llm if isinstance(client_llm, ClientLlmConfig) else None
    p = merged_cursor_launch_params(cl)
    if not p["api_key"]:
        raise RuntimeError("Cursor 模式下缺少 API Key")
    url = f"{_base_url(p['base_url'])}/v0/agents/{agent_id}"
    r = _http_request(
        "GET",
        url,
        headers=_get_headers(p["api_key"]),
        timeout=_POLL_REQUEST_TIMEOUT,
        context="get_agent_status",
    )
    _raise_for_status_with_body(r, "get_agent_status")
    return r.json()


def get_agent_conversation(agent_id: str, *, client_llm: Optional[Any] = None) -> list:
    from src.core.llm_factory import ClientLlmConfig, merged_cursor_launch_params

    cl = client_llm if isinstance(client_llm, ClientLlmConfig) else None
    p = merged_cursor_launch_params(cl)
    if not p["api_key"]:
        raise RuntimeError("Cursor 模式下缺少 API Key")
    url = f"{_base_url(p['base_url'])}/v0/agents/{agent_id}/conversation"
    r = _http_request(
        "GET",
        url,
        headers=_get_headers(p["api_key"]),
        timeout=_POLL_REQUEST_TIMEOUT,
        context="get_agent_conversation",
    )
    _raise_for_status_with_body(r, "get_agent_conversation")
    data = r.json()
    return data.get("messages") or []


def poll_until_finished(
    agent_id: str,
    poll_interval: float = 5.0,
    timeout: float = 3600,
    *,
    client_llm: Optional[Any] = None,
) -> str:
    deadline = time.monotonic() + timeout
    last_log = 0.0
    last_status = ""
    while time.monotonic() < deadline:
        status_data = get_agent_status(agent_id, client_llm=client_llm)
        status = (status_data.get("status") or "").upper()
        last_status = status
        if status in ("FINISHED", "FAILED", "STOPPED", "ERROR"):
            return status
        now = time.monotonic()
        if now - last_log >= 60:
            waited = timeout - max(0.0, deadline - now)
            _log.info(
                "Cursor Agent %s 仍在运行 status=%s，已等待 %.0fs / %.0fs",
                agent_id,
                status,
                waited,
                timeout,
            )
            last_log = now
        time.sleep(max(2.0, float(poll_interval)))
    return last_status or "TIMEOUT"


def get_last_assistant_reply(agent_id: str, *, client_llm: Optional[Any] = None) -> Optional[str]:
    messages = get_agent_conversation(agent_id, client_llm=client_llm)
    texts = []
    for m in messages:
        if (m.get("type") or "") == "assistant_message" and m.get("text"):
            texts.append(m["text"].strip())
    if not texts:
        return None
    return "\n\n".join(texts)


def _cursor_poll_timeout_seconds(explicit: Optional[float]) -> float:
    if explicit is not None and float(explicit) > 0:
        return float(explicit)
    try:
        v = int(getattr(settings, "cursor_agent_timeout_seconds", 0) or 0)
        if v > 0:
            return float(max(300, min(v, 6 * 3600)))
    except Exception:
        pass
    return 3600.0


def harvest_agent_reply(agent_id: str, *, client_llm: Optional[Any] = None) -> tuple[str, Optional[str]]:
    """取云端 Agent 最终状态与助手回复（不创建新任务）。"""
    status_data = get_agent_status(agent_id, client_llm=client_llm)
    status = (status_data.get("status") or "").upper() or "UNKNOWN"
    reply = None
    if status in ("FINISHED", "FAILED", "STOPPED", "ERROR", "TIMEOUT"):
        try:
            reply = get_last_assistant_reply(agent_id, client_llm=client_llm)
        except Exception as e:
            _log.warning("取 Cursor Agent 对话失败 agent_id=%s: %s", agent_id, e)
    else:
        try:
            reply = get_last_assistant_reply(agent_id, client_llm=client_llm)
        except Exception:
            reply = None
    return status, reply


def complete_task(
    prompt_text: str,
    poll_interval: float = 5.0,
    timeout: Optional[float] = None,
    *,
    client_llm: Optional[Any] = None,
) -> str:
    from src.core.llm_factory import ClientLlmConfig, get_request_client_llm

    cl = client_llm if isinstance(client_llm, ClientLlmConfig) else None
    if cl is None or not cl.has_any():
        cl = get_request_client_llm()
    wait_s = _cursor_poll_timeout_seconds(timeout)
    agent_id = launch_agent(prompt_text, client_llm=cl)
    _log.info("Cursor Agent 已创建 agent_id=%s，最长等待 %ss（超时仍会尝试取回已完成回复）", agent_id, int(wait_s))
    status = poll_until_finished(
        agent_id,
        poll_interval=max(2.0, float(poll_interval)),
        timeout=wait_s,
        client_llm=cl,
    )
    if status not in ("FINISHED", "FAILED", "STOPPED", "ERROR"):
        harvested_status, harvested_reply = harvest_agent_reply(agent_id, client_llm=cl)
        if harvested_reply and harvested_status == "FINISHED":
            _log.warning(
                "本地轮询结束但云端已完成，已取回回复 agent_id=%s chars=%s",
                agent_id,
                len(harvested_reply),
            )
            return harvested_reply
        if harvested_reply:
            _log.warning(
                "轮询超时但已有部分/完整回复，按已完成使用 agent_id=%s status=%s",
                agent_id,
                harvested_status,
            )
            return harvested_reply
        status = harvested_status or status or "TIMEOUT"
    if status != "FINISHED":
        raise RuntimeError(
            f"Cursor Agent 未完成: status={status}, agent_id={agent_id}。"
            "云端任务可能仍在运行并继续扣额度，请勿立即重新提交同一任务；"
            "到 Cursor Dashboard → Cloud Agents 查看该 ID。"
            " 若稍后显示 FINISHED，重启服务后再生成前可先用该 agent_id 取回结果。"
        )
    reply = get_last_assistant_reply(agent_id, client_llm=cl)
    if not reply:
        raise RuntimeError(f"Cursor Agent 未返回对话内容, agent_id={agent_id}")
    return reply
