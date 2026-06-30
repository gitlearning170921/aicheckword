"""
调用 Cursor Cloud Agents API：发起任务、轮询完成、取回助手回复文本。
支持请求级凭据（``ClientLlmConfig`` / aiword Header）与 ``settings`` 合并。
"""

from __future__ import annotations

import base64
import time
from typing import Any, Callable, Optional, TypeVar

import httpx

from config import settings

_T = TypeVar("_T")

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


def _http_client(timeout: float = 60) -> httpx.Client:
    """统一创建 httpx 客户端。支持：关闭 SSL 校验、显式/环境变量代理。"""
    try:
        from config.cursor_overrides import build_llm_httpx_client

        t = timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(
            timeout, connect=min(45.0, max(15.0, timeout * 0.25))
        )
        return build_llm_httpx_client(timeout=t)
    except Exception:
        return httpx.Client(timeout=timeout, verify=True, trust_env=True, http2=False)


def _is_transient_network_error(exc: BaseException) -> bool:
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
    return RuntimeError(
        f"无法连接 Cursor API（{_base_url(base)}）：{exc}。"
        "请检查：1) 本机/代理能否稳定访问 api.cursor.com（Clash 建议开 TUN 或固定节点）；"
        "2) .env 中 HTTP_PROXY/LLM_HTTP_PROXY 是否与 Clash 端口一致；"
        "3) 证书异常时可试侧栏「不校验 SSL」。"
        " 若为偶发 SSL EOF，程序已自动重试；仍失败多为代理链路不稳，请换节点或稍后重试。"
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
            if attempt < max_attempts and _is_transient_network_error(e):
                time.sleep(_CURSOR_HTTP_RETRY_BASE_SEC * attempt)
                continue
            raise
    if last is not None:
        raise last
    raise RuntimeError(f"{context}: 请求失败")


def _http_request(
    method: str,
    url: str,
    *,
    headers: dict,
    json: Optional[dict] = None,
    timeout: float = 60,
    context: str = "",
) -> httpx.Response:
    def _do() -> httpx.Response:
        with _http_client(timeout=timeout) as client:
            if method.upper() == "POST":
                return client.post(url, json=json, headers=headers)
            return client.get(url, headers=headers)

    try:
        return _with_http_retry(_do, context=context or url)
    except _TRANSIENT_NET_EXC as e:
        raise _cursor_connect_runtime_error(url, e) from e


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
    r = _http_request(
        "POST",
        url,
        json=payload,
        headers=_get_headers(p["api_key"]),
        timeout=90,
        context="launch_agent",
    )
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
    poll_interval: float = 2.0,
    timeout: float = 300,
    *,
    client_llm: Optional[Any] = None,
) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status_data = get_agent_status(agent_id, client_llm=client_llm)
        status = (status_data.get("status") or "").upper()
        if status in ("FINISHED", "FAILED", "STOPPED"):
            return status
        time.sleep(poll_interval)
    return "TIMEOUT"


def get_last_assistant_reply(agent_id: str, *, client_llm: Optional[Any] = None) -> Optional[str]:
    messages = get_agent_conversation(agent_id, client_llm=client_llm)
    texts = []
    for m in messages:
        if (m.get("type") or "") == "assistant_message" and m.get("text"):
            texts.append(m["text"].strip())
    if not texts:
        return None
    return "\n\n".join(texts)


def complete_task(
    prompt_text: str,
    poll_interval: float = 2.0,
    timeout: float = 600,
    *,
    client_llm: Optional[Any] = None,
) -> str:
    from src.core.llm_factory import ClientLlmConfig, get_request_client_llm

    cl = client_llm if isinstance(client_llm, ClientLlmConfig) else None
    if cl is None or not cl.has_any():
        cl = get_request_client_llm()
    agent_id = launch_agent(prompt_text, client_llm=cl)
    status = poll_until_finished(agent_id, poll_interval=poll_interval, timeout=timeout, client_llm=cl)
    if status != "FINISHED":
        raise RuntimeError(f"Cursor Agent 未完成: status={status}, agent_id={agent_id}")
    reply = get_last_assistant_reply(agent_id, client_llm=cl)
    if not reply:
        raise RuntimeError(f"Cursor Agent 未返回对话内容, agent_id={agent_id}")
    return reply
