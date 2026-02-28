"""
调用 Cursor Cloud Agents API：发起任务、轮询完成、取回助手回复文本。
文档: https://cursor.com/docs/cloud-agent/api/overview
认证: Dashboard → Integrations 创建 API Key，Basic Auth (key: 空密码)。
"""

import base64
import time
from typing import Optional

import httpx

from config import settings


def _auth_header() -> str:
    """Basic Auth: API_KEY 作为用户名，密码为空"""
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


def launch_agent(prompt_text: str) -> str:
    """
    发起一个 Cloud Agent 任务，返回 agent id。
    要求 settings 中已配置 cursor_api_key、cursor_repository。
    """
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
    with httpx.Client(timeout=60) as client:
        r = client.post(url, json=payload, headers=_get_headers())
        r.raise_for_status()
        data = r.json()
    return data["id"]


def get_agent_status(agent_id: str) -> dict:
    """获取 Agent 状态"""
    url = f"{_base_url()}/v0/agents/{agent_id}"
    with httpx.Client(timeout=30) as client:
        r = client.get(url, headers=_get_headers())
        r.raise_for_status()
        return r.json()


def get_agent_conversation(agent_id: str) -> list:
    """获取 Agent 对话记录，返回 messages 列表"""
    url = f"{_base_url()}/v0/agents/{agent_id}/conversation"
    with httpx.Client(timeout=30) as client:
        r = client.get(url, headers=_get_headers())
        r.raise_for_status()
        data = r.json()
    return data.get("messages") or []


def poll_until_finished(
    agent_id: str,
    poll_interval: float = 2.0,
    timeout: float = 300,
) -> str:
    """
    轮询直到 Agent 状态为 FINISHED 或 FAILED。
    返回状态字符串。
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status_data = get_agent_status(agent_id)
        status = (status_data.get("status") or "").upper()
        if status in ("FINISHED", "FAILED", "STOPPED"):
            return status
        time.sleep(poll_interval)
    return "TIMEOUT"


def get_last_assistant_reply(agent_id: str) -> Optional[str]:
    """
    从 Agent 对话中取最后一条助手回复的 text。
    若有多条 assistant_message，拼接最后一条（或全部）文本返回。
    """
    messages = get_agent_conversation(agent_id)
    texts = []
    for m in messages:
        if (m.get("type") or "") == "assistant_message" and m.get("text"):
            texts.append(m["text"].strip())
    if not texts:
        return None
    return "\n\n".join(texts)


def complete_task(prompt_text: str, poll_interval: float = 2.0, timeout: float = 300) -> str:
    """
    发起 Cursor Agent 任务，轮询完成，并返回助手的回复文本。
    用于「只输出文本、不修改仓库」的提示（如文档审核 JSON 输出）。
    """
    agent_id = launch_agent(prompt_text)
    status = poll_until_finished(agent_id, poll_interval=poll_interval, timeout=timeout)
    if status != "FINISHED":
        raise RuntimeError(f"Cursor Agent 未完成: status={status}, agent_id={agent_id}")
    reply = get_last_assistant_reply(agent_id)
    if not reply:
        raise RuntimeError(f"Cursor Agent 未返回对话内容, agent_id={agent_id}")
    return reply
