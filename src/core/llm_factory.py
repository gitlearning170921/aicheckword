"""
统一创建 Chat 模型：根据 config.settings.provider 选择 Ollama / OpenAI / Gemini / 通义 / 文心 / 零一万物 / DeepSeek / Claude 等。
零一万物、DeepSeek 使用 OpenAI 兼容接口，走 ChatOpenAI + base_url；Claude 走 Anthropic Messages API。
"""

import os
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional

import httpx

from config import settings
from config.cursor_overrides import get_llm_verify_ssl, get_llm_trust_env


@dataclass
class ClientLlmConfig:
    """单次请求级 LLM 凭据（如 aiword 用户自带 Key）。禁止写入日志或 DB。

    Cursor Cloud Agents：``api_key`` 为 Cursor Dashboard API Key；``base_url`` 为 Cursor API Base；
    ``cursor_repository`` / ``cursor_ref`` 对应 GitHub 仓库与分支（与系统设置字段语义一致）。

    ``personal_keys_only``：为 True 时 **API Key 仅用请求内字段**，不回退到 ``settings``（如 aiword 页面2个人配置）。
    """

    api_key: str = ""
    base_url: str = ""
    model: str = ""
    cursor_repository: str = ""
    cursor_ref: str = ""
    personal_keys_only: bool = False

    def has_any(self) -> bool:
        return bool(
            (self.api_key or "").strip()
            or (self.base_url or "").strip()
            or (self.model or "").strip()
            or (self.cursor_repository or "").strip()
            or (self.cursor_ref or "").strip()
        )


_request_client_llm: ContextVar[Optional["ClientLlmConfig"]] = ContextVar(
    "request_client_llm", default=None
)


def get_request_client_llm() -> Optional[ClientLlmConfig]:
    """集成 job 线程内当前请求级 LLM 凭据（由 ``activate_request_client_llm`` 注入）。"""
    return _request_client_llm.get()


@contextmanager
def activate_request_client_llm(
    client_llm: Optional[ClientLlmConfig],
) -> Iterator[None]:
    """在 integration 后台 job 内激活个人/请求级 Key，供 ``invoke_chat_direct`` 等自动拾取。"""
    token: Token = _request_client_llm.set(client_llm)
    try:
        yield
    finally:
        _request_client_llm.reset(token)


def bind_request_client_llm(client_llm: Optional[ClientLlmConfig]) -> Token:
    """非 ``with`` 场景下绑定 job 级凭据；须配对 ``unbind_request_client_llm``。"""
    return _request_client_llm.set(client_llm)


def unbind_request_client_llm(token: Token) -> None:
    _request_client_llm.reset(token)


def normalize_api_key_plain(plain: str) -> str:
    s = (plain or "").replace("\r", "").replace("\n", "")
    for zw in ("\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u00a0"):
        s = s.replace(zw, "")
    s = s.strip()
    while s.lower().startswith("bearer "):
        s = s[7:].strip()
    return s


def normalize_openai_compatible_base_url(provider: str, base: str) -> str:
    p = (provider or "").strip().lower()
    b = (base or "").strip().rstrip("/")
    if not b:
        return ""
    if p == "deepseek" and "api.deepseek.com" in b.lower():
        rest = b.split("://", 1)[-1].split("/", 1)
        tail = rest[1] if len(rest) > 1 else ""
        if not tail.startswith("v1"):
            return b + "/v1"
    return b


def client_llm_from_mapping(raw: Any) -> Optional[ClientLlmConfig]:
    if not isinstance(raw, dict):
        return None
    prov = str(raw.get("provider") or raw.get("_provider") or "").strip().lower()
    bu = str(raw.get("base_url") or "").strip()
    return ClientLlmConfig(
        api_key=normalize_api_key_plain(str(raw.get("api_key") or "")),
        base_url=normalize_openai_compatible_base_url(prov or "deepseek", bu) if bu else bu,
        model=str(raw.get("model") or "").strip(),
        cursor_repository=str(raw.get("cursor_repository") or "").strip(),
        cursor_ref=str(raw.get("cursor_ref") or "").strip(),
        personal_keys_only=bool(raw.get("personal_keys_only")),
    )


def resolve_client_llm(
    *,
    explicit: Optional[ClientLlmConfig] = None,
    review_context: Optional[Mapping[str, Any]] = None,
) -> Optional[ClientLlmConfig]:
    """合并显式参数、ContextVar 与 review_context['_client_llm']。"""
    # personal_keys_only=False 且显式传入时，表示「走系统 settings」，即使 api_key 等字段为空，
    # 也不应被 ContextVar 中的个人 Key 覆盖（llm-key-test 系统 Key 探测依赖此语义）。
    if explicit is not None and (explicit.has_any() or not explicit.personal_keys_only):
        return explicit
    ctx = get_request_client_llm()
    if ctx is not None and ctx.has_any():
        return ctx
    if review_context:
        mapped = client_llm_from_mapping(review_context.get("_client_llm"))
        if mapped is not None and mapped.has_any():
            return mapped
    return None


def _auth_error_message(*, provider: str, client_llm: Optional[ClientLlmConfig]) -> str:
    labels = {
        "deepseek": "DeepSeek",
        "openai": "OpenAI",
        "lingyi": "零一万物",
        "tongyi": "通义千问",
        "claude": "Claude (Anthropic)",
        "ollama": "Ollama",
    }
    label = labels.get((provider or "").strip().lower(), provider or "LLM")
    msg = f"{label} API Key 鉴权失败（HTTP 401）。"
    if client_llm and getattr(client_llm, "personal_keys_only", False):
        msg += (
            " 当前为个人 Key 模式：Key 无效或与 aicheckword 侧栏配置不一致。"
            "个人 Key 仅替换 API Key，Base URL/模型沿用 aicheckword 系统设置；"
            "请在 aiword 初稿页点「测试 Key」验证。"
        )
    else:
        msg += " 请检查 aicheckword 系统配置中的 API Key，或在 aiword 保存个人 Key 后重试。"
    return msg


def _raise_for_llm_http_status(
    response: httpx.Response,
    *,
    provider: str,
    client_llm: Optional[ClientLlmConfig],
) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            raise RuntimeError(
                _auth_error_message(provider=provider, client_llm=client_llm)
            ) from exc
        body = ""
        try:
            body = (exc.response.text or "").strip()[:800]
        except Exception:
            pass
        label = {
            "deepseek": "DeepSeek",
            "openai": "OpenAI",
            "lingyi": "零一万物",
            "tongyi": "通义千问",
            "claude": "Claude",
            "ollama": "Ollama",
        }.get((provider or "").strip().lower(), provider or "LLM")
        msg = f"{label} 聊天接口失败（HTTP {exc.response.status_code}）"
        if body:
            msg += f"：{body}"
        msg += "。请核对 API Key、Base URL、模型名；中转服务须支持 /chat/completions。"
        raise RuntimeError(msg) from exc


_DEFAULT_MODEL_BY_PROVIDER: dict[str, str] = {
    "deepseek": "deepseek-chat",
    "openai": "gpt-4o-mini",
    "lingyi": "yi-lightning",
    "tongyi": "qwen-plus",
    "claude": "claude-sonnet-4-20250514",
    "ollama": "qwen2.5",
}


def default_model_for_provider(provider: Optional[str]) -> str:
    """按 provider 取默认模型；仅当与全局 settings.provider 一致时才用 settings.llm_model。"""
    p = (provider or settings.provider or "deepseek").strip().lower()
    if p == (settings.provider or "").strip().lower() and (settings.llm_model or "").strip():
        cand = settings.llm_model.strip()
        el = cand.lower()
        if p == "openai" and any(x in el for x in ("qwen", "deepseek", "yi-", "claude", "gemini", "ernie")) and "gpt" not in el:
            return _DEFAULT_MODEL_BY_PROVIDER.get("openai", "gpt-4o-mini")
        if p == "tongyi" and any(x in el for x in ("deepseek", "gpt", "yi-lightning", "claude", "gemini")):
            return _DEFAULT_MODEL_BY_PROVIDER.get("tongyi", "qwen-plus")
        return cand
    return _DEFAULT_MODEL_BY_PROVIDER.get(p, settings.llm_model or "qwen-plus")


def resolve_model_for_provider(
    provider: Optional[str],
    *,
    model: Optional[str] = None,
    client_llm: Optional[ClientLlmConfig] = None,
) -> str:
    """解析本次请求实际使用的模型名，避免「选通义却带 deepseek-chat 模型名」。"""
    p = (provider or settings.provider or "").strip().lower()
    explicit = ((client_llm.model if client_llm else "") or model or "").strip()
    if explicit:
        el = explicit.lower()
        if p == "tongyi" and any(x in el for x in ("deepseek", "gpt", "yi-lightning", "claude", "gemini")):
            return default_model_for_provider("tongyi")
        if p == "deepseek" and "qwen" in el and "deepseek" not in el:
            return default_model_for_provider("deepseek")
        if p == "openai" and any(x in el for x in ("qwen", "deepseek", "yi-", "claude", "gemini", "ernie")) and "gpt" not in el:
            return default_model_for_provider("openai")
        if p == "claude" and any(x in el for x in ("deepseek", "gpt", "qwen", "yi-", "gemini")) and "claude" not in el:
            return default_model_for_provider("claude")
        return explicit
    return default_model_for_provider(p)


def merged_cursor_launch_params(client_llm: Optional[ClientLlmConfig] = None) -> dict[str, str]:
    """合并请求级 ClientLlmConfig 与 ``settings``，得到 Cursor Agents 所需的 Key / Base / 仓库 / ref。"""
    cl = client_llm
    strict = bool(cl and cl.personal_keys_only)
    if strict:
        ak = ((cl.api_key if cl else "") or "").strip()
    else:
        ak = ((cl.api_key if cl else "") or "").strip() or (settings.cursor_api_key or "").strip()
    base = ((cl.base_url if cl else "") or "").strip() or (settings.cursor_api_base or "https://api.cursor.com").strip()
    repo = ((cl.cursor_repository if cl else "") or "").strip() or (settings.cursor_repository or "").strip()
    ref = ((cl.cursor_ref if cl else "") or "").strip() or (settings.cursor_ref or "main").strip()
    return {
        "api_key": ak,
        "base_url": base.rstrip("/"),
        "repository": repo,
        "ref": ref or "main",
    }


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


def _claude_api_key() -> str:
    return (settings.claude_api_key or "").strip()


def _claude_base_url() -> str:
    return (settings.claude_base_url or "https://api.anthropic.com").strip().rstrip("/")


def _claude_http_client(*, timeout: Optional[httpx.Timeout] = None) -> httpx.Client:
    t = timeout or httpx.Timeout(600.0, connect=45.0)
    return httpx.Client(verify=get_llm_verify_ssl(), trust_env=get_llm_trust_env(), timeout=t)


def _create_claude_chat_llm(temperature: float = 0.1):
    api_key = _claude_api_key()
    if not api_key:
        raise RuntimeError("Claude 模式下请先配置 ANTHROPIC_API_KEY 或 claude_api_key（.env 或侧栏）")
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        raise RuntimeError("请安装 langchain-anthropic：pip install langchain-anthropic") from e
    base_url = _claude_base_url()
    kwargs: dict[str, Any] = {
        "model": settings.llm_model or "claude-sonnet-4-20250514",
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": 8192,
    }
    if base_url and base_url.rstrip("/") != "https://api.anthropic.com":
        kwargs["base_url"] = base_url
    try:
        kwargs["http_client"] = _claude_http_client(timeout=httpx.Timeout(600.0, connect=45.0))
        return ChatAnthropic(**kwargs)
    except TypeError:
        kwargs.pop("http_client", None)
        return ChatAnthropic(**kwargs)


def _invoke_claude_messages(
    prompt_text: str,
    *,
    temperature: float,
    model: str,
    api_key: str,
    base_url: str,
    client_llm: Optional[ClientLlmConfig],
) -> str:
    if not api_key:
        raise RuntimeError("Claude 模式下请先配置 API Key 或在请求中传入 X-Client-Llm-Api-Key")
    url = f"{base_url.rstrip('/')}/v1/messages"
    payload = {
        "model": model or "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt_text}],
    }
    with _claude_http_client(timeout=httpx.Timeout(600.0, connect=45.0)) as client:
        r = client.post(
            url,
            json=payload,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=600,
        )
        _raise_for_llm_http_status(r, provider="claude", client_llm=client_llm)
        data = r.json()
    blocks = data.get("content") or []
    parts = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            txt = block.get("text")
            if isinstance(txt, str) and txt.strip():
                parts.append(txt.strip())
    if parts:
        return "\n".join(parts)
    raise RuntimeError("Claude Messages API 返回无文本 content")


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

    if p == "claude":
        return _create_claude_chat_llm(temperature)

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
        "claude": "Claude (Anthropic)",
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
    model: Optional[str] = None,
    client_llm: Optional[ClientLlmConfig] = None,
) -> str:
    """
    直接调用当前配置的聊天接口，返回助手回复内容。
    用于多文档一致性等场景，避免 LangChain 将 content 当模板解析导致 {\"category\"} 报错。
    支持：openai / deepseek / lingyi（OpenAI 兼容）、claude（Anthropic Messages）、ollama、tongyi（DashScope Generation）。
    provider: 若传入（如从 Streamlit current_provider），则优先使用，否则用 settings.provider。
    client_llm: 若传入且含 api_key/base_url/model 等，则对应字段覆盖 settings（用于集成 API / 用户自带 Key）。
    """
    p = (provider or settings.provider or "").strip().lower()
    cl = resolve_client_llm(explicit=client_llm)
    personal_only = bool(cl and getattr(cl, "personal_keys_only", False))
    m = resolve_model_for_provider(
        p,
        model=model,
        client_llm=None if personal_only else cl,
    )

    if p == "tongyi":
        # 通义走 DashScope 官方域名，与 OpenAI 兼容服务的 Base URL 无关；侧栏也无「通义 Base URL」项。
        if cl and getattr(cl, "personal_keys_only", False):
            api_key = normalize_api_key_plain((cl.api_key if cl else "") or "")
        else:
            api_key = normalize_api_key_plain(
                ((cl.api_key if cl else "") or (settings.dashscope_api_key or ""))
            )
        if not api_key:
            raise RuntimeError("通义模式下请先配置 DASHSCOPE_API_KEY或在请求中传入 X-Client-Llm-Api-Key")
        try:
            from dashscope import Generation
        except ImportError as e:
            raise RuntimeError("请安装 dashscope：pip install dashscope") from e
        try:
            resp = Generation.call(
                model=m or "qwen-plus",
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

    if p == "claude":
        if cl and getattr(cl, "personal_keys_only", False):
            api_key = normalize_api_key_plain((cl.api_key if cl else "") or "")
            raw_base = _claude_base_url()
        else:
            api_key = normalize_api_key_plain(((cl.api_key if cl else "") or _claude_api_key()))
            raw_base = ((cl.base_url if cl else "") or _claude_base_url()).strip().rstrip("/")
        return _invoke_claude_messages(
            prompt_text,
            temperature=temperature,
            model=m,
            api_key=api_key,
            base_url=raw_base or "https://api.anthropic.com",
            client_llm=cl,
        )

    with _openai_http_client() as client:
        if p in ("openai", "deepseek", "lingyi"):
            if cl and getattr(cl, "personal_keys_only", False):
                api_key = normalize_api_key_plain((cl.api_key if cl else "") or "")
                try:
                    from config.settings import openai_form_base_url_default_from_settings

                    raw_base = openai_form_base_url_default_from_settings(p).strip()
                except Exception:
                    raw_base = _openai_compatible_base_url(p).strip()
            else:
                api_key = normalize_api_key_plain(
                    ((cl.api_key if cl else "") or _openai_compatible_api_key(p))
                )
                raw_base = ((cl.base_url if cl else "") or _openai_compatible_base_url(p)).strip()
            base_url = (
                normalize_openai_compatible_base_url(p, raw_base) or raw_base
            ).rstrip("/")
            if not api_key:
                raise RuntimeError(f"{p} 模式下请先配置 API Key或在请求中传入 X-Client-Llm-Api-Key")
            url = f"{base_url}/chat/completions"
            payload = {
                "model": m or ("deepseek-chat" if p == "deepseek" else "gpt-4o-mini"),
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": temperature,
            }
            req_timeout = 600.0 if p in ("openai", "deepseek") else 300.0
            try:
                r = client.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    timeout=req_timeout,
                )
            except httpx.TimeoutException as e:
                _lbl = {"openai": "OpenAI", "deepseek": "DeepSeek", "lingyi": "零一万物"}.get(p, p)
                raise RuntimeError(
                    f"{_lbl} 聊天请求超时（{int(req_timeout)}s）。"
                    "文档过长或网关较慢时可尝试缩短文档、换更快模型，或检查 Base URL 网络。"
                ) from e
            except (httpx.ConnectError, httpx.ReadError, OSError) as e:
                el = str(e).lower()
                if "10054" in str(e) or "connection" in el or "reset" in el or "eof" in el:
                    _lbl = {"openai": "OpenAI", "deepseek": "DeepSeek", "lingyi": "零一万物"}.get(p, p)
                    raise RuntimeError(
                        f"{_lbl} 无法连接 {base_url}（{e}）。"
                        "国内/公司网直连 api.openai.com 常被重置；请改用**中转 Base URL**，"
                        "或在侧栏尝试「不使用系统代理」「不校验 SSL」。"
                    ) from e
                raise
            _raise_for_llm_http_status(r, provider=p, client_llm=cl)
            data = r.json()
            choice = (data.get("choices") or [None])[0]
            if not choice:
                raise RuntimeError("聊天接口返回无 choices")
            return (choice.get("message") or {}).get("content") or ""

        if p == "ollama":
            base_url = (
                ((cl.base_url if cl else "") or (settings.ollama_base_url or "http://localhost:11434")).strip().rstrip("/")
            )
            url = f"{base_url}/api/chat"
            payload = {
                "model": m or "qwen2.5",
                "messages": [{"role": "user", "content": prompt_text}],
                "stream": False,
            }
            r = client.post(url, json=payload, timeout=300)
            _raise_for_llm_http_status(r, provider=p, client_llm=cl)
            data = r.json()
            msg = data.get("message") or {}
            return msg.get("content") or ""

    # 其他 provider 暂不实现，由调用方处理
    raise RuntimeError(
        f"invoke_chat_direct 暂不支持 provider={p}，请使用 Cursor / OpenAI / DeepSeek / 零一 / Claude / Ollama / 通义"
    )
