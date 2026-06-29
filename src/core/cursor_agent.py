"""
调用 Cursor Cloud Agents API：发起任务、轮询完成、取回助手回复文本。
支持请求级凭据（``ClientLlmConfig`` / aiword Header）与 ``settings`` 合并。
"""

from __future__ import annotations

import base64
import time
from typing import Any, Optional

import httpx

from config import settings


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
    """统一创建 httpx 客户端。支持：关闭 SSL 校验、绕过系统代理（代理导致 SSL EOF 时）"""
    try:
        from config.cursor_overrides import get_cursor_verify_ssl, get_cursor_trust_env

        verify = get_cursor_verify_ssl()
        trust_env = get_cursor_trust_env()
    except Exception:
        verify = True
        trust_env = True
    return httpx.Client(timeout=timeout, verify=verify, trust_env=trust_env)


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
    with _http_client(timeout=90) as client:
        try:
            r = client.post(url, json=payload, headers=_get_headers(p["api_key"]))
        except (httpx.ConnectError, httpx.ReadError, OSError) as e:
            el = str(e).lower()
            if "10054" in str(e) or "connection" in el or "reset" in el or "eof" in el or "ssl" in el:
                raise RuntimeError(
                    f"无法连接 Cursor API（{_base_url(p['base_url'])}）：{e}。"
                    "请检查：1) 本机/代理能否访问 api.cursor.com；"
                    "2) 侧栏「不使用系统代理」是否与 Clash 规则匹配（Cursor 需走代理时可取消勾选）；"
                    "3) 证书异常时可试「不校验 SSL」。"
                ) from e
            raise
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
    with _http_client(timeout=_POLL_REQUEST_TIMEOUT) as client:
        r = client.get(url, headers=_get_headers(p["api_key"]))
        _raise_for_status_with_body(r, "get_agent_status")
        return r.json()


def get_agent_conversation(agent_id: str, *, client_llm: Optional[Any] = None) -> list:
    from src.core.llm_factory import ClientLlmConfig, merged_cursor_launch_params

    cl = client_llm if isinstance(client_llm, ClientLlmConfig) else None
    p = merged_cursor_launch_params(cl)
    if not p["api_key"]:
        raise RuntimeError("Cursor 模式下缺少 API Key")
    url = f"{_base_url(p['base_url'])}/v0/agents/{agent_id}/conversation"
    with _http_client(timeout=_POLL_REQUEST_TIMEOUT) as client:
        r = client.get(url, headers=_get_headers(p["api_key"]))
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
