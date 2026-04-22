"""
统一创建 Chat 模型：根据 config.settings.provider 选择 Ollama / OpenAI / Gemini / 通义 / 文心 / 零一万物 / DeepSeek 等。
零一万物、DeepSeek 使用 OpenAI 兼容接口，走 ChatOpenAI + base_url。
"""

import os
from typing import Optional

import httpx

from config import settings
from config.cursor_overrides import get_llm_verify_ssl, get_llm_trust_env


def _ensure_tiktoken_no_proxy():
    """避免 tiktoken 从 openaipublic.blob 下载 cl100k_base 时走代理导致 ProxyError/SSLEOFError。"""
    host = "openaipublic.blob.core.windows.net"
    for key in ("NO_PROXY", "no_proxy"):
        cur = os.environ.get(key, "")
        if host not in cur:
            os.environ[key] = f"{cur},{host}".lstrip(",")


def _openai_http_client(*, timeout: Optional[httpx.Timeout] = None) -> httpx.Client:
    """供 ChatOpenAI 使用的 httpx 客户端，应用「不校验 SSL」「不使用系统代理」通用配置。"""
    # 长审核/大上下文时默认超时过短易被误判为 Connection error；DeepSeek 等在工厂内单独加长
    t = timeout or httpx.Timeout(600.0, connect=45.0)
    return httpx.Client(verify=get_llm_verify_ssl(), trust_env=get_llm_trust_env(), timeout=t)


def create_chat_llm(temperature: float = 0.1):
    """
    创建 LangChain Chat 实例。Cursor 模式不应调用本函数（由 complete_task 处理）。
    """
    p = (settings.provider or "").strip().lower()

    if p == "ollama":
        from langchain_ollama import ChatOllama
        # Ollama 使用 client_kwargs 传入 verify/trust_env，与通用「不校验 SSL」「不使用系统代理」一致
        client_kwargs = {"verify": get_llm_verify_ssl(), "trust_env": get_llm_trust_env()}
        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            client_kwargs=client_kwargs,
        )

    if p == "cursor":
        raise RuntimeError("Cursor 模式下请使用 complete_task，不应调用 create_chat_llm")

    # OpenAI 兼容：openai / deepseek / lingyi（零一万物）
    if p in ("openai", "deepseek", "lingyi"):
        _ensure_tiktoken_no_proxy()
        from langchain_openai import ChatOpenAI
        api_key = _openai_compatible_api_key(p)
        base_url = _openai_compatible_base_url(p)
        if not api_key:
            raise RuntimeError(f"{p} 模式下请先配置 API Key（.env 或侧栏）")
        # DeepSeek 等大上下文审核耗时更长；过短易在客户端表现为连接/读超时类错误
        req_timeout = 600.0 if p == "deepseek" else 180.0
        # DeepSeek 在客户端层过多重试会叠峰请求、放大 CPU/内存与限流风险，略降重试由应用层单次重试兜底
        _mr = 2 if p == "deepseek" else 5
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            http_client=_openai_http_client(timeout=httpx.Timeout(req_timeout, connect=45.0)),
            request_timeout=req_timeout,
            max_retries=_mr,
        )

    # Google Gemini
    if p == "gemini":
        api_key = (settings.gemini_api_key or settings.google_api_key or "").strip()
        if not api_key:
            raise RuntimeError("Gemini 模式下请先配置 GEMINI_API_KEY 或 GOOGLE_API_KEY")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise RuntimeError("请安装 langchain-google-genai：pip install langchain-google-genai")
        return ChatGoogleGenerativeAI(
            model=settings.llm_model or "gemini-1.5-flash",
            google_api_key=api_key,
            temperature=temperature,
        )

    # 阿里通义（DashScope）
    if p == "tongyi":
        api_key = (settings.dashscope_api_key or "").strip()
        if not api_key:
            raise RuntimeError("通义模式下请先配置 DASHSCOPE_API_KEY")
        try:
            from langchain_community.chat_models import ChatTongyi
        except ImportError:
            raise RuntimeError("请安装 dashscope 与 langchain-community：pip install dashscope")
        return ChatTongyi(
            model=settings.llm_model or "qwen-plus",
            api_key=api_key,
            temperature=temperature,
        )

    # 百度文心（千帆）
    if p == "baidu":
        ak = (settings.qianfan_ak or "").strip()
        sk = (settings.qianfan_sk or "").strip()
        if not ak or not sk:
            raise RuntimeError("文心模式下请先配置 QIANFAN_AK 与 QIANFAN_SK")
        try:
            from langchain_community.chat_models import ChatBaiduQianfan
        except ImportError:
            raise RuntimeError("请安装 qianfan：pip install qianfan")
        return ChatBaiduQianfan(
            model=settings.llm_model or "ERNIE-Bot-4",
            qianfan_ak=ak,
            qianfan_sk=sk,
            temperature=temperature,
        )

    # 默认按 OpenAI 兼容处理
    _ensure_tiktoken_no_proxy()
    from langchain_openai import ChatOpenAI
    if not settings.openai_api_key:
        raise RuntimeError("OpenAI 模式下请先配置 API Key")
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=temperature,
        http_client=_openai_http_client(timeout=httpx.Timeout(180.0, connect=45.0)),
        request_timeout=180,
        max_retries=5,
    )


def _openai_compatible_api_key(provider: str) -> str:
    if provider == "deepseek":
        return (settings.deepseek_api_key or settings.openai_api_key or "").strip()
    if provider == "lingyi":
        return (settings.lingyi_api_key or settings.openai_api_key or "").strip()
    return (settings.openai_api_key or "").strip()


def _openai_compatible_base_url(provider: str) -> str:
    if provider == "deepseek":
        return (settings.deepseek_base_url or "https://api.deepseek.com/v1").strip()
    if provider == "lingyi":
        return (settings.lingyi_base_url or "https://api.lingyiwanwu.com/v1").strip()
    return (settings.openai_base_url or "https://api.openai.com/v1").strip()


def provider_needs_openai_compatible_form(provider: str) -> bool:
    """侧栏是否展示 OpenAI 式 Key + Base URL 表单（含 DeepSeek / 零一）。"""
    p = (provider or "").strip().lower()
    return p in ("openai", "deepseek", "lingyi")


def provider_display_name(provider: str) -> str:
    m = {
        "ollama": "Ollama (本地)",
        "openai": "OpenAI",
        "cursor": "Cursor Agent",
        "gemini": "Google Gemini",
        "tongyi": "阿里通义千问",
        "baidu": "百度文心一言",
        "lingyi": "零一万物",
        "deepseek": "DeepSeek",
    }
    return m.get((provider or "").lower(), provider or "unknown")


def _dashscope_generation_text(resp) -> str:
    """从 DashScope Generation.call 的响应中取助手文本；失败则抛 RuntimeError。"""
    sc = int(getattr(resp, "status_code", 0) or 0)
    if sc != 200:
        raise RuntimeError(getattr(resp, "message", None) or str(resp))
    code = (getattr(resp, "code", None) or "").strip()
    if code:
        raise RuntimeError(f"{code}: {getattr(resp, 'message', '') or '请求失败'}")
    out = getattr(resp, "output", None)
    if not out:
        return ""
    tx = getattr(out, "text", None)
    if isinstance(tx, str) and tx.strip():
        return tx.strip()
    choices = getattr(out, "choices", None) or []
    if choices:
        msg = getattr(choices[0], "message", None)
        if msg is not None:
            c = getattr(msg, "content", None)
            if isinstance(c, str):
                return c.strip()
    return ""


def invoke_chat_direct(
    prompt_text: str,
    temperature: float = 0.1,
    provider: Optional[str] = None,
) -> str:
    """
    直接调用当前配置的聊天接口，返回助手回复内容。
    用于多文档一致性等场景，避免 LangChain 将 content 当模板解析导致 {\"category\"} 报错。
    支持：openai / deepseek / lingyi（OpenAI 兼容）、ollama、tongyi（DashScope Generation）。
    provider: 若传入（如从 Streamlit current_provider），则优先使用，否则用 settings.provider。
    """
    p = (provider or settings.provider or "").strip().lower()

    if p == "tongyi":
        # 通义走 DashScope 官方域名，与 OpenAI 兼容服务的 Base URL 无关；侧栏也无「通义 Base URL」项。
        api_key = (settings.dashscope_api_key or "").strip()
        if not api_key:
            raise RuntimeError("通义模式下请先配置 DASHSCOPE_API_KEY")
        try:
            from dashscope import Generation
        except ImportError as e:
            raise RuntimeError("请安装 dashscope：pip install dashscope") from e
        try:
            resp = Generation.call(
                model=settings.llm_model or "qwen-plus",
                messages=[{"role": "user", "content": prompt_text}],
                api_key=api_key,
                temperature=temperature,
                result_format="message",
            )
        except OSError as e:
            raise RuntimeError(
                "通义 DashScope 网络连接失败（与是否填写 OpenAI Base URL 无关）。"
                "请检查：本机/服务器能否访问阿里云、防火墙与代理、DASHSCOPE_API_KEY 是否有效；"
                "若走 HTTPS 拦截代理，可尝试系统信任证书或在可直连环境运行。"
            ) from e
        except Exception as e:
            el = str(e).lower()
            if any(x in el for x in ("connection", "timeout", "10054", "ssl", "certificate")):
                raise RuntimeError(
                    "通义 DashScope 请求异常（多为网络/SSL/超时）。通义无需配置 Base URL；"
                    "请核对 Key、出网策略及本机时间同步。"
                ) from e
            raise
        return _dashscope_generation_text(resp)

    with _openai_http_client() as client:
        if p in ("openai", "deepseek", "lingyi"):
            api_key = _openai_compatible_api_key(p)
            base_url = _openai_compatible_base_url(p).rstrip("/")
            if not api_key:
                raise RuntimeError(f"{p} 模式下请先配置 API Key")
            url = f"{base_url}/chat/completions"
            payload = {
                "model": settings.llm_model or ("deepseek-chat" if p == "deepseek" else "gpt-4o-mini"),
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": temperature,
            }
            r = client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=300,
            )
            r.raise_for_status()
            data = r.json()
            choice = (data.get("choices") or [None])[0]
            if not choice:
                raise RuntimeError("聊天接口返回无 choices")
            return (choice.get("message") or {}).get("content") or ""

        if p == "ollama":
            base_url = (settings.ollama_base_url or "http://localhost:11434").rstrip("/")
            url = f"{base_url}/api/chat"
            payload = {
                "model": settings.llm_model or "qwen2.5",
                "messages": [{"role": "user", "content": prompt_text}],
                "stream": False,
            }
            r = client.post(url, json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()
            msg = data.get("message") or {}
            return msg.get("content") or ""

    # 其他 provider 暂不实现，由调用方处理
    raise RuntimeError(
        f"invoke_chat_direct 暂不支持 provider={p}，请使用 Cursor / OpenAI / DeepSeek / 零一 / Ollama / 通义"
    )
