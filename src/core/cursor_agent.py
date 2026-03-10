"""
调用 Cursor Cloud Agents API：发起任务、轮询完成、取回助手回复文本。
"""

import base64
import time
from typing import Optional

import httpx

from config import settings


def _auth_header() -> str:
    raw = f"{settings.cursor_api_key}:"
    return "Basic " + base64.b64encode(raw.encode()).decode()


def _get_headers() -> dict:
    return {
        "Authorization": _auth_header(),
        "Content-Type": "application/json",
    }


def _base_url() -> str:
    base = (settings.cursor_api_base or "https://api.cursor.com").rstrip("/")
    return base


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


def launch_agent(prompt_text: str) -> str:
    if not settings.cursor_api_key or not settings.cursor_repository:
        raise RuntimeError("Cursor 模式下请配置 API Key 和 GitHub 仓库地址（Cursor Dashboard → Integrations）")
    url = f"{_base_url()}/v0/agents"
    payload = {
        "prompt": {"text": prompt_text},
        "source": {
            "repository": settings.cursor_repository.strip(),
            "ref": (settings.cursor_ref or "main").strip(),
        },
        "target": {"autoCreatePr": False},
    }
    with _http_client(timeout=90) as client:
        r = client.post(url, json=payload, headers=_get_headers())
        r.raise_for_status()
        data = r.json()
    return data["id"]


# 轮询状态/拉取对话时单次请求超时（秒），避免多文档等长任务时 read timeout
_POLL_REQUEST_TIMEOUT = 120


def get_agent_status(agent_id: str) -> dict:
    url = f"{_base_url()}/v0/agents/{agent_id}"
    with _http_client(timeout=_POLL_REQUEST_TIMEOUT) as client:
        r = client.get(url, headers=_get_headers())
        r.raise_for_status()
        return r.json()


def get_agent_conversation(agent_id: str) -> list:
    url = f"{_base_url()}/v0/agents/{agent_id}/conversation"
    with _http_client(timeout=_POLL_REQUEST_TIMEOUT) as client:
        r = client.get(url, headers=_get_headers())
        r.raise_for_status()
        data = r.json()
    return data.get("messages") or []


def poll_until_finished(
    agent_id: str,
    poll_interval: float = 2.0,
    timeout: float = 300,
) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status_data = get_agent_status(agent_id)
        status = (status_data.get("status") or "").upper()
        if status in ("FINISHED", "FAILED", "STOPPED"):
            return status
        time.sleep(poll_interval)
    return "TIMEOUT"


def get_last_assistant_reply(agent_id: str) -> Optional[str]:
    messages = get_agent_conversation(agent_id)
    texts = []
    for m in messages:
        if (m.get("type") or "") == "assistant_message" and m.get("text"):
            texts.append(m["text"].strip())
    if not texts:
        return None
    return "\n\n".join(texts)


def complete_task(prompt_text: str, poll_interval: float = 2.0, timeout: float = 600) -> str:
    agent_id = launch_agent(prompt_text)
    status = poll_until_finished(agent_id, poll_interval=poll_interval, timeout=timeout)
    if status != "FINISHED":
        raise RuntimeError(f"Cursor Agent 未完成: status={status}, agent_id={agent_id}")
    reply = get_last_assistant_reply(agent_id)
    if not reply:
        raise RuntimeError(f"Cursor Agent 未返回对话内容, agent_id={agent_id}")
    return reply
