"""软件版本公开发布时间候选检索（供 aiword 版本任务清单集成调用）。

优先按注册国家对应应用市场/商店发布页检索；网页失败时再批量走 LLM 推断。
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, datetime, timedelta, timezone
from html import unescape
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

import httpx

logger = logging.getLogger(__name__)

DATE_PATTERNS = (
    re.compile(r"\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b"),
    re.compile(r"\b(20\d{2})年(\d{1,2})月(\d{1,2})日\b"),
)

VERSION_RE = re.compile(r"^[Vv]?\s*(\d+)\.(\d+)\.(\d+)\.(\d+)\s*$")

_DDG_URL_TEMPLATES = (
    "https://html.duckduckgo.com/html/?q={query}",
    "https://lite.duckduckgo.com/lite/?q={query}",
)
_DDG_INTER_REQUEST_SLEEP_SEC = 0.85
# 单版本最多查询次数（避免「只点一个版本却打出多条搜索」）
_MAX_QUERIES_PER_VERSION = 2

# App Store（iTunes Search API）：比 Bing/DDG HTML 更稳，不被验证码拦截
_ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
_ITUNES_LOOKUP_URL = "https://itunes.apple.com/lookup"

LlmTextFn = Callable[[str], str]

# 注册国家 → 应用市场检索关键词（与常见 SaMD/独立软件分发渠道对齐）
_COUNTRY_MARKET_HINTS: dict[str, tuple[str, ...]] = {
    "中国": ("App Store", "应用宝", "华为应用市场", "小米应用商店", "版本更新", "更新日志"),
    "china": ("App Store", "应用宝", "华为应用市场", "小米应用商店", "版本更新"),
    "cn": ("App Store", "应用宝", "华为应用市场", "小米应用商店"),
    "美国": ("App Store", "Google Play", "release notes", "what's new", "version history"),
    "usa": ("App Store", "Google Play", "release notes", "version history"),
    "us": ("App Store", "Google Play", "release notes", "version history"),
    "欧盟": ("App Store", "Google Play", "release notes", "version history", "CE"),
    "欧洲": ("App Store", "Google Play", "release notes", "version history"),
    "eu": ("App Store", "Google Play", "release notes", "version history"),
    "英国": ("App Store", "Google Play", "release notes", "version history"),
    "uk": ("App Store", "Google Play", "release notes", "version history"),
    "日本": ("App Store", "Google Play", "リリースノート", "バージョン"),
    "japan": ("App Store", "Google Play", "リリースノート"),
    "jp": ("App Store", "Google Play", "リリースノート"),
    "澳大利亚": ("App Store", "Google Play", "release notes"),
    "australia": ("App Store", "Google Play", "release notes"),
    "巴西": ("App Store", "Google Play", "notas da versão"),
    "加拿大": ("App Store", "Google Play", "release notes"),
}


def _fmt_date(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def _safe_date(y: int, m: int, d: int) -> Optional[str]:
    try:
        return _fmt_date(date(y, m, d))
    except Exception:
        return None


def _extract_dates(text: str) -> list[str]:
    out: list[str] = []
    for pattern in DATE_PATTERNS:
        for m in pattern.finditer(text or ""):
            parsed = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if parsed:
                out.append(parsed)
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq


def _normalize_version(raw: str) -> str:
    text = (raw or "").strip()
    m = VERSION_RE.match(text)
    if not m:
        raise ValueError(f"版本号格式错误：{raw}（应为 X.Y.Z.B）")
    return ".".join(str(int(m.group(i))) for i in range(1, 5))


def _parse_version_chain(
    from_version: str,
    to_version: str,
    intermediate_versions: Optional[list[str]] = None,
) -> list[str]:
    chain_raw = [from_version]
    for item in intermediate_versions or []:
        s = (item or "").strip()
        if s:
            chain_raw.append(s)
    chain_raw.append(to_version)
    return [_normalize_version(v) for v in chain_raw]


# 注册路径/证照简称 → 检索用地区（项目里常见填 CE/FDA/NMPA）
_REGULATORY_REGION_ALIASES: dict[str, str] = {
    "ce": "欧盟",
    "ce标记": "欧盟",
    "ce mark": "欧盟",
    "ce-mark": "欧盟",
    "cemark": "欧盟",
    "mdr": "欧盟",
    "mdd": "欧盟",
    "eu": "欧盟",
    "eu mdr": "欧盟",
    "欧洲": "欧盟",
    "欧盟": "欧盟",
    "fda": "美国",
    "510k": "美国",
    "510(k)": "美国",
    "pma": "美国",
    "usa": "美国",
    "us": "美国",
    "美国": "美国",
    "nmpa": "中国",
    "cfda": "中国",
    "nmpa/cfda": "中国",
    "china": "中国",
    "cn": "中国",
    "prc": "中国",
    "国内": "中国",
    "中国": "中国",
}


def normalize_registration_region(registration_country: str) -> str:
    """把 CE/FDA/NMPA 等注册路径映射到检索地区；无法识别则原样返回。"""
    raw = (registration_country or "").strip()
    if not raw:
        return ""
    lower = raw.casefold().replace("／", "/").replace("－", "-")
    # 精确
    if lower in _REGULATORY_REGION_ALIASES:
        return _REGULATORY_REGION_ALIASES[lower]
    # 去空格后再精确
    compact = re.sub(r"\s+", "", lower)
    if compact in _REGULATORY_REGION_ALIASES:
        return _REGULATORY_REGION_ALIASES[compact]
    # 含证照简称（如「CE获证」「FDA 510k」「NMPA三类」）
    for code, region in (
        ("nmpa", "中国"),
        ("cfda", "中国"),
        ("510(k)", "美国"),
        ("510k", "美国"),
        ("fda", "美国"),
        ("mdr", "欧盟"),
        ("mdd", "欧盟"),
        ("ce", "欧盟"),
    ):
        if code in lower:
            return region
    return raw


def _is_china_country(registration_country: str) -> bool:
    region = normalize_registration_region(registration_country)
    key = (region or "").strip().casefold()
    if not key:
        return False
    markers = ("中国", "china", "cn", "prc", "mainland", "国内")
    return any(m in key for m in markers)


def _market_hints_for_country(registration_country: str) -> list[str]:
    region = normalize_registration_region(registration_country)
    key = (region or "").strip()
    if not key:
        return ["App Store", "Google Play", "release notes", "version history"]
    lower = key.casefold()
    for name, hints in _COUNTRY_MARKET_HINTS.items():
        if name.casefold() == lower or name in key or key in name:
            return list(hints)
    # 通用兜底：地区名 + 主流双商店（非默认国内渠道）
    return [key, "App Store", "Google Play", "release notes", "version history"]


def _bing_mkt_for_country(registration_country: str) -> tuple[str, str]:
    """返回 (mkt, setlang)。"""
    region = normalize_registration_region(registration_country)
    key = (region or "").strip().casefold()
    if _is_china_country(region):
        return "zh-CN", "zh-CN"
    mapping = (
        (("美国", "usa", "us", "united states"), "en-US", "en"),
        (("英国", "uk", "united kingdom", "gb"), "en-GB", "en"),
        (("欧盟", "欧洲", "eu", "europe", "德国", "de", "france", "法国", "fr"), "en-US", "en"),
        (("日本", "japan", "jp"), "ja-JP", "ja"),
        (("澳大利亚", "australia", "au"), "en-AU", "en"),
        (("加拿大", "canada", "ca"), "en-CA", "en"),
        (("巴西", "brazil", "br"), "pt-BR", "pt"),
    )
    for keys, mkt, lang in mapping:
        if any(k in key for k in keys):
            return mkt, lang
    return "en-US", "en"


def _quote_product_for_query(product_name: str) -> str:
    """产品名加引号做精确短语检索；内部引号剥离。"""
    product = (product_name or "").strip().strip('"').strip("'")
    if not product:
        return ""
    return f'"{product}"'


def _build_search_queries(
    *,
    product_name: str,
    version: str,
    registration_country: str,
    max_queries: int = _MAX_QUERIES_PER_VERSION,
) -> list[str]:
    """按注册地区构造检索词：产品名精确短语 + 版本 + 地区/商店渠道。"""
    product = (product_name or "").strip()
    raw_country = (registration_country or "").strip()
    region = normalize_registration_region(raw_country)
    markets = _market_hints_for_country(region)
    quoted = _quote_product_for_query(product)
    # 无产品名时不瞎搜「软件」，避免漂到无关国内页
    base = f"{quoted} {version}".strip() if quoted else version
    china = _is_china_country(region)
    release_kw = "发布" if china else "release"
    notes_kw = "更新日志" if china else "release notes"
    market0 = markets[0] if markets else ("应用市场" if china else "App Store")
    queries: list[str] = [
        f"{base} {release_kw}",
        f"{base} {market0}",
    ]
    if region:
        queries.append(f"{base} {region} {notes_kw}")
    else:
        queries.append(f"{base} {notes_kw}")
    # 非中国：英文发布说明 + 映射后地区（CE→欧盟）优先
    if not china:
        queries.insert(0, f"{base} {notes_kw}")
        if region:
            queries.insert(1, f"{base} {region}")
        # 保留原始 CE/FDA 字样，便于命中合规公告页
        if raw_country and raw_country.casefold() != region.casefold():
            queries.insert(2, f"{base} {raw_country}")
    seen: set[str] = set()
    out: list[str] = []
    limit = max(1, int(max_queries or _MAX_QUERIES_PER_VERSION))
    for q in queries:
        q = " ".join(str(q or "").split())
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(q)
        if len(out) >= limit:
            break
    return out


def _web_search_headers(*, registration_country: str = "") -> dict[str, str]:
    _, setlang = _bing_mkt_for_country(registration_country)
    if _is_china_country(registration_country):
        accept_lang = "zh-CN,zh;q=0.9,en;q=0.5"
    elif setlang.startswith("ja"):
        accept_lang = "ja,en;q=0.8"
    else:
        accept_lang = "en-US,en;q=0.9"
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": accept_lang,
        # 避免代理复用坏连接引发 10054
        "Connection": "close",
    }


def _is_proxy_or_ssl_error(exc: BaseException) -> bool:
    lowered = (str(exc) or "").lower()
    return any(
        key in lowered
        for key in (
            "proxyerror",
            "unable to connect to proxy",
            "ssleoferror",
            "eof occurred in violation of protocol",
            "certificate",
            "ssl",
            "proxy",
            "10054",
            "connection reset",
            "connection aborted",
            "forcibly closed",
            "remotely closed",
        )
    )


def _llm_shared_httpx_timeout(timeout: float) -> httpx.Timeout:
    """与 cursor_agent._http_client 相同的 Timeout 构造。"""
    return httpx.Timeout(timeout, connect=min(45.0, max(15.0, timeout * 0.25)))


def _duckduckgo_fetch_html(
    query: str,
    timeout: float,
    *,
    registration_country: str = "",
) -> tuple[str, dict[str, Any]]:
    """复用 Cursor/LLM 代理客户端拉取 DDG HTML（不再单独要求配置）。"""
    from config.cursor_overrides import (
        build_llm_httpx_client,
        get_llm_http_proxy,
        get_llm_verify_ssl,
        llm_httpx_client_kwargs,
    )

    proxy = get_llm_http_proxy()
    verify_default = bool(get_llm_verify_ssl())
    urls = [tpl.format(query=quote(query)) for tpl in _DDG_URL_TEMPLATES]
    headers = _web_search_headers(registration_country=registration_country)
    t = _llm_shared_httpx_timeout(timeout)

    # 与 Cursor 共用：1) mounts 客户端 2) for_url 显式代理 3) SSL 关闭回退
    attempt_builders: list[tuple[str, Any]] = [
        ("llm_shared_mounts", lambda _url: build_llm_httpx_client(timeout=t)),
        (
            "llm_shared_for_url",
            lambda url: httpx.Client(
                **{
                    **llm_httpx_client_kwargs(for_url=url),
                    "timeout": t,
                    "follow_redirects": True,
                }
            ),
        ),
    ]
    if verify_default:
        attempt_builders.append(
            (
                "llm_shared_for_url_insecure",
                lambda url: httpx.Client(
                    **{
                        **llm_httpx_client_kwargs(for_url=url),
                        "timeout": t,
                        "verify": False,
                        "follow_redirects": True,
                    }
                ),
            )
        )

    diagnostics: dict[str, Any] = {
        "proxyConfigured": bool(proxy),
        "proxyMode": "llm_shared",
        "proxyNormalized": proxy or "",
        "attempts": [],
        "networkError": None,
        "url": urls[0] if urls else "",
        "failureHint": None,
    }
    last_err: Optional[str] = None
    for url in urls:
        for label, builder in attempt_builders:
            for retry_i in range(2):
                attempt_label = label if retry_i == 0 else f"{label}_retry{retry_i}"
                try:
                    with builder(url) as client:
                        resp = client.get(url, headers=headers, follow_redirects=True)
                        resp.raise_for_status()
                        html = resp.text
                    diagnostics["url"] = url
                    diagnostics["httpStatus"] = resp.status_code
                    diagnostics["attemptUsed"] = attempt_label
                    diagnostics["attempts"].append(
                        {"label": attempt_label, "ok": True, "status": resp.status_code}
                    )
                    return html, diagnostics
                except Exception as exc:
                    err = str(exc)
                    last_err = err
                    diagnostics["attempts"].append(
                        {"label": attempt_label, "ok": False, "error": err[:240]}
                    )
                    # 10054 / 连接重置：换新连接再试一次
                    if retry_i == 0 and (
                        "10054" in err or "connection reset" in err.lower() or "aborted" in err.lower()
                    ):
                        time.sleep(0.4)
                        continue
                    if not _is_proxy_or_ssl_error(exc) and "timeout" not in err.lower():
                        break
                    break

    diagnostics["networkError"] = last_err or "DuckDuckGo 请求失败"
    diagnostics["failureHint"] = (
        "DuckDuckGo 检索失败（已复用 Cursor/LLM 同一代理配置）。将尝试国内 Bing 回退。"
    )
    return "", diagnostics


def _unwrap_ddg_href(href: str) -> str:
    text = (href or "").strip()
    if not text:
        return ""
    if text.startswith("//"):
        text = "https:" + text
    try:
        parsed = urlparse(text)
    except Exception:
        return text
    host = (parsed.hostname or "").lower()
    if "duckduckgo.com" in host and ("/l/" in (parsed.path or "") or "uddg=" in (parsed.query or "")):
        qs = parse_qs(parsed.query)
        for key in ("uddg", "u"):
            vals = qs.get(key) or []
            if vals:
                return unquote(vals[0])
    return text


def _parse_duckduckgo_html(html: str) -> tuple[list[dict[str, str]], str]:
    patterns: list[tuple[str, re.Pattern[str]]] = [
        (
            "primary",
            re.compile(
                r'<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
                r'<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
                re.S,
            ),
        ),
        (
            "fallback_class",
            re.compile(
                r'<a[^>]*class="[^"]*result[^"]*"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
                re.S,
            ),
        ),
    ]
    out: list[dict[str, str]] = []
    parser = "none"
    for parser_name, block_re in patterns:
        for m in block_re.finditer(html):
            href = _unwrap_ddg_href(unescape(re.sub(r"\s+", " ", m.group("href") or "").strip()))
            title = unescape(re.sub(r"<.*?>", "", m.group("title") or "").strip())
            snippet = ""
            if "snippet" in m.groupdict():
                snippet = unescape(re.sub(r"<.*?>", "", m.group("snippet") or "").strip())
            if not href:
                continue
            out.append({"url": href, "title": title, "snippet": snippet})
            if len(out) >= 10:
                break
        if out:
            parser = parser_name
            break

    if not out:
        link_re = re.compile(r'href="((?:https?:)?//[^"]+|https?://[^"]+)"[^>]*>([^<]{4,120})<', re.I)
        for m in link_re.finditer(html):
            href = _unwrap_ddg_href(unescape(m.group(1).strip()))
            title = unescape(re.sub(r"<.*?>", "", m.group(2) or "").strip())
            if not href or "duckduckgo.com" in href:
                continue
            out.append({"url": href, "title": title, "snippet": ""})
            if len(out) >= 8:
                break
        if out:
            parser = "link_fallback"
    return out, parser


def _duckduckgo_search_with_diagnostics(
    query: str,
    timeout: float = 15.0,
    *,
    registration_country: str = "",
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    started = time.time()
    html, diagnostics = _duckduckgo_fetch_html(
        query, timeout, registration_country=registration_country
    )
    diagnostics.setdefault("httpStatus", None)
    diagnostics.setdefault("htmlBytes", 0)
    diagnostics.setdefault("parser", "none")
    diagnostics.setdefault("rawHits", 0)
    diagnostics.setdefault("durationMs", 0)
    diagnostics["engine"] = "duckduckgo"
    logger.info(
        "release-date-search: duckduckgo query=%s proxy=%s attempt=%s",
        query,
        diagnostics.get("proxyMode"),
        diagnostics.get("attemptUsed") or "failed",
    )
    if diagnostics.get("networkError") or not html:
        diagnostics["durationMs"] = int((time.time() - started) * 1000)
        logger.warning(
            "release-date-search duckduckgo failed query=%s err=%s",
            query,
            diagnostics.get("networkError"),
        )
        return [], diagnostics

    diagnostics["htmlBytes"] = len(html.encode("utf-8", errors="ignore"))
    out, parser = _parse_duckduckgo_html(html)
    diagnostics["parser"] = parser
    diagnostics["rawHits"] = len(out)
    diagnostics["durationMs"] = int((time.time() - started) * 1000)
    if not out and ("anomaly" in html.lower() or "captcha" in html.lower() or len(html) < 800):
        diagnostics["failureHint"] = (
            "DuckDuckGo 返回异常页/验证码或空页，将尝试 Bing 回退"
        )
    logger.info(
        "release-date-search: duckduckgo raw_hits=%d parser=%s proxy=%s attempt=%s",
        len(out),
        parser,
        diagnostics.get("proxyMode"),
        diagnostics.get("attemptUsed"),
    )
    return out, diagnostics


def _parse_bing_html(html: str) -> tuple[list[dict[str, str]], str]:
    block_re = re.compile(
        r'<li[^>]*class="b_algo"[^>]*>.*?<h2[^>]*>\s*<a[^>]+href="(?P<href>[^"]+)"[^>]*>'
        r"(?P<title>.*?)</a>.*?(?:<p[^>]*>|<div[^>]*class=\"b_caption\"[^>]*>)"
        r"(?P<snippet>.*?)(?:</p>|</div>)",
        re.S | re.I,
    )
    out: list[dict[str, str]] = []
    for m in block_re.finditer(html or ""):
        href = unescape(re.sub(r"\s+", " ", m.group("href") or "").strip())
        title = unescape(re.sub(r"<.*?>", "", m.group("title") or "").strip())
        snippet = unescape(re.sub(r"<.*?>", "", m.group("snippet") or "").strip())
        if not href or not title:
            continue
        out.append({"url": href, "title": title, "snippet": snippet})
        if len(out) >= 10:
            break
    if out:
        return out, "bing_b_algo"
    # 宽松回退：抓取结果区链接
    link_re = re.compile(
        r'<h2[^>]*>\s*<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>',
        re.S | re.I,
    )
    for m in link_re.finditer(html or ""):
        href = unescape(m.group(1).strip())
        title = unescape(re.sub(r"<.*?>", "", m.group(2) or "").strip())
        if not href or not title or "bing.com" in href:
            continue
        out.append({"url": href, "title": title, "snippet": ""})
        if len(out) >= 8:
            break
    return out, ("bing_h2" if out else "none")


def _bing_urls_for_country(query: str, registration_country: str) -> list[str]:
    mkt, setlang = _bing_mkt_for_country(registration_country)
    q = quote(query)
    if _is_china_country(registration_country):
        return [
            f"https://cn.bing.com/search?q={q}&setlang=zh-CN",
            f"https://www.bing.com/search?q={q}&setlang=zh-CN&mkt=zh-CN",
        ]
    # 非中国：只用国际 Bing + 对应 mkt，避免 cn.bing 漂移到国内结果
    return [
        f"https://www.bing.com/search?q={q}&setlang={setlang}&mkt={mkt}",
        f"https://www.bing.com/search?q={q}&setlang=en&mkt=en-US",
    ]


def _parse_bing_rss(xml: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for block in re.findall(r"<item>(.*?)</item>", xml or "", re.I | re.S):
        title_m = re.search(r"<title>(.*?)</title>", block, re.I | re.S)
        link_m = re.search(r"<link>(.*?)</link>", block, re.I | re.S)
        desc_m = re.search(r"<description>(.*?)</description>", block, re.I | re.S)
        title = unescape(re.sub(r"<.*?>", "", title_m.group(1) if title_m else "")).strip()
        link = unescape(re.sub(r"<.*?>", "", link_m.group(1) if link_m else "")).strip()
        snippet = unescape(re.sub(r"<.*?>", "", desc_m.group(1) if desc_m else "")).strip()
        if not title or not link:
            continue
        # Bing RSS 被风控时会灌一堆无关 Microsoft/Google 账号页
        low = f"{title} {link}".casefold()
        if any(
            x in low
            for x in (
                "myapplications.microsoft.com",
                "accounts.google.com",
                "myactivity.google.com",
                "mychart.org",
            )
        ):
            continue
        out.append({"url": link, "title": title, "snippet": snippet})
        if len(out) >= 10:
            break
    return out


def _is_bing_challenge_html(html: str) -> bool:
    low = (html or "").lower()
    if "b_algo" in low:
        return False
    return ("captcha" in low) or ("challenge" in low) or ("anomaly" in low)


def _bing_search_with_diagnostics(
    query: str,
    timeout: float = 15.0,
    *,
    registration_country: str = "",
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """按注册国家选择 Bing：优先 RSS（少验证码），HTML 失败则快速放弃重复重试。"""
    from config.cursor_overrides import build_llm_httpx_client, get_llm_verify_ssl, llm_httpx_client_kwargs

    started = time.time()
    china = _is_china_country(registration_country)
    mkt, setlang = _bing_mkt_for_country(registration_country)
    q = quote(query)
    urls = [
        f"https://www.bing.com/search?q={q}&format=rss&setlang={setlang}&mkt={mkt}",
        *_bing_urls_for_country(query, registration_country)[:1],
    ]
    headers = _web_search_headers(registration_country=registration_country)
    t = _llm_shared_httpx_timeout(timeout)
    verify = bool(get_llm_verify_ssl())
    diagnostics: dict[str, Any] = {
        "engine": "bing",
        "proxyMode": "direct_cn" if china else "llm_shared_intl",
        "registrationCountry": registration_country or "",
        "attempts": [],
        "networkError": None,
        "url": urls[0] if urls else "",
        "httpStatus": None,
        "htmlBytes": 0,
        "parser": "none",
        "rawHits": 0,
        "durationMs": 0,
    }
    last_err: Optional[str] = None
    for url in urls:
        is_rss = "format=rss" in url
        is_cn_host = "cn.bing.com" in url
        builders: list[tuple[str, Any]] = []
        if is_cn_host or china:
            builders.append(
                (
                    "bing_direct",
                    lambda _u=url: httpx.Client(
                        timeout=t,
                        verify=verify,
                        trust_env=False,
                        http2=False,
                        follow_redirects=True,
                        headers=headers,
                    ),
                )
            )
        builders.append(
            (
                "bing_llm_shared",
                lambda u=url: httpx.Client(
                    **{
                        **llm_httpx_client_kwargs(for_url=u),
                        "timeout": t,
                        "follow_redirects": True,
                        "headers": headers,
                    }
                ),
            )
        )
        if not is_rss:
            builders.append(
                ("bing_llm_mounts", lambda _u=url: build_llm_httpx_client(timeout=t)),
            )
        for label, builder in builders:
            try:
                with builder() as client:
                    resp = client.get(url, headers=headers, follow_redirects=True)
                    resp.raise_for_status()
                    body = resp.text
                if is_rss:
                    out = _parse_bing_rss(body)
                    parser = "bing_rss"
                else:
                    if _is_bing_challenge_html(body):
                        last_err = "Bing 返回验证码/挑战页（自动化抓取受限，与浏览器手工不同）"
                        diagnostics["attempts"].append(
                            {"label": label, "ok": False, "error": last_err}
                        )
                        # 验证码页不必换 client 再试同 URL
                        break
                    out, parser = _parse_bing_html(body)
                diagnostics["url"] = url
                diagnostics["httpStatus"] = resp.status_code
                diagnostics["attemptUsed"] = label
                diagnostics["htmlBytes"] = len(body.encode("utf-8", errors="ignore"))
                diagnostics["parser"] = parser
                diagnostics["rawHits"] = len(out)
                diagnostics["durationMs"] = int((time.time() - started) * 1000)
                diagnostics["attempts"].append(
                    {"label": label, "ok": True, "status": resp.status_code, "hits": len(out)}
                )
                if out:
                    logger.info(
                        "release-date-search: bing country=%s raw_hits=%d parser=%s attempt=%s",
                        registration_country or "-",
                        len(out),
                        parser,
                        label,
                    )
                    return out, diagnostics
                last_err = "Bing 页面未解析到有效结果"
                if is_rss:
                    break
            except Exception as exc:
                err = str(exc)
                last_err = err
                diagnostics["attempts"].append({"label": label, "ok": False, "error": err[:240]})
    diagnostics["networkError"] = last_err or "Bing 请求失败"
    diagnostics["durationMs"] = int((time.time() - started) * 1000)
    diagnostics["failureHint"] = "Bing 检索未成功，将尝试其它引擎或请手动填写"
    logger.warning("release-date-search bing failed query=%s err=%s", query, last_err)
    return [], diagnostics


def _web_search_with_diagnostics(
    query: str,
    timeout: float = 15.0,
    *,
    registration_country: str = "",
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """按注册国家检索：中国优先 cn.Bing；其它优先国际 Bing，再 DDG。"""
    bing_results, bing_diag = _bing_search_with_diagnostics(
        query, timeout=timeout, registration_country=registration_country
    )
    if bing_results:
        return bing_results, bing_diag
    ddg_results, ddg_diag = _duckduckgo_search_with_diagnostics(
        query, timeout=timeout, registration_country=registration_country
    )
    merged = {
        **bing_diag,
        "engine": ddg_diag.get("engine") if ddg_results else bing_diag.get("engine"),
        "fallback": {
            "duckduckgo": {
                "rawHits": ddg_diag.get("rawHits"),
                "attemptUsed": ddg_diag.get("attemptUsed"),
                "networkError": ddg_diag.get("networkError"),
                "parser": ddg_diag.get("parser"),
            }
        },
    }
    if ddg_results:
        merged["attemptUsed"] = ddg_diag.get("attemptUsed")
        merged["parser"] = ddg_diag.get("parser")
        merged["rawHits"] = ddg_diag.get("rawHits")
        merged["networkError"] = None
        merged["failureHint"] = None
        merged["httpStatus"] = ddg_diag.get("httpStatus")
        merged["url"] = ddg_diag.get("url")
        merged["proxyMode"] = ddg_diag.get("proxyMode")
        return ddg_results, merged
    merged["networkError"] = ddg_diag.get("networkError") or bing_diag.get("networkError")
    merged["failureHint"] = (
        ddg_diag.get("failureHint")
        or bing_diag.get("failureHint")
        or "网页检索失败，请手动填写发布时间"
    )
    return [], merged


_CN_HOST_MARKERS = (
    "baidu.com",
    "zhihu.com",
    "csdn.net",
    "jianshu.com",
    "juejin.cn",
    "cnblogs.com",
    "qq.com",
    "163.com",
    "sohu.com",
    "sina.com",
    "huawei.com",
    "xiaomi.com",
    "yingyongbao",
)


def _result_relevance_score(
    row: dict[str, str],
    *,
    product_name: str,
    version: str,
    registration_country: str,
) -> int:
    """按产品名命中/国家相关性打分；非中国项目压低国内站。"""
    title = str(row.get("title") or "")
    snippet = str(row.get("snippet") or "")
    url = str(row.get("url") or "").lower()
    text = f"{title} {snippet}".strip()
    text_l = text.casefold()
    score = 0
    product = (product_name or "").strip()
    if product:
        pl = product.casefold()
        if pl in text_l or pl in url:
            score += 20
        else:
            # 产品名未出现：大幅降权，但仍保留无日期以外的候选窗口
            score -= 15
            # 多词产品取最长 token
            tokens = [t for t in re.split(r"[\s\-_/]+", product) if len(t) >= 2]
            if any(t.casefold() in text_l for t in tokens):
                score += 10
    ver = (version or "").strip()
    if ver and ver in text:
        score += 8
    china = _is_china_country(registration_country)
    if not china:
        if any(m in url for m in _CN_HOST_MARKERS):
            score -= 12
        # 大量汉字且无产品名：更像国内噪音页
        cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
        if cjk >= 18 and product and product.casefold() not in text_l:
            score -= 8
        country = (registration_country or "").strip()
        if country and country.casefold() in text_l:
            score += 6
    else:
        if any(m in url for m in ("apple.com", "play.google.com", "huawei.com", "mi.com")):
            score += 4
    return score


def _build_candidates_from_results(
    *,
    version: str,
    results: list[dict[str, str]],
    include_skipped: bool = False,
    product_name: str = "",
    registration_country: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    product = (product_name or "").strip()
    for row in results:
        text = f"{row.get('title', '')} {row.get('snippet', '')}"
        dates = _extract_dates(text)
        if not dates:
            skipped.append(
                {
                    "title": str(row.get("title") or "")[:120],
                    "reason": "snippet中未匹配到日期",
                }
            )
            continue
        rel = _result_relevance_score(
            row,
            product_name=product,
            version=version,
            registration_country=registration_country,
        )
        # 填了产品名却完全不相关：丢掉，避免误用国内无关发布时间
        if product and rel < -5:
            skipped.append(
                {
                    "title": str(row.get("title") or "")[:120],
                    "reason": f"与产品/注册国相关性过低(score={rel})",
                }
            )
            continue
        conf = "medium" if rel >= 18 else "low"
        candidates.append(
            {
                "version": version,
                "date": dates[0],
                "sourceUrl": row.get("url") or "",
                "sourceTitle": row.get("title") or "",
                "snippet": row.get("snippet") or "",
                "confidence": conf,
                "sourceKind": "web",
                "relevanceScore": rel,
            }
        )
    dedup: dict[str, dict[str, Any]] = {}
    for c in candidates:
        prev = dedup.get(c["date"])
        if not prev or int(c.get("relevanceScore") or 0) > int(prev.get("relevanceScore") or 0):
            dedup[c["date"]] = c
    ordered = sorted(
        dedup.values(),
        key=lambda x: (int(x.get("relevanceScore") or 0), x.get("date") or ""),
        reverse=True,
    )
    extraction: dict[str, Any] = {
        "rowsScanned": len(results),
        "rowsWithDate": len(candidates),
        "rowsSkippedNoDate": len(skipped),
        "candidateCount": len(ordered),
        "productName": product,
        "registrationCountry": registration_country or "",
    }
    if include_skipped:
        extraction["skippedSamples"] = skipped[:5]
    if not results:
        extraction["failureHint"] = "应用市场/网页检索未解析到结果（网络受限或页面结构变更）"
    elif results and not ordered:
        if product:
            extraction["failureHint"] = (
                f"已命中网页，但未找到与产品「{product}」及注册国家匹配且含日期的条目"
            )
        else:
            extraction["failureHint"] = "已命中搜索结果，但标题/摘要中未提取到发布日期"
    return ordered[:5], extraction


def _parse_llm_json_array(text: str) -> list[dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
    except Exception:
        pass
    m = re.search(r"\[[\s\S]*\]", raw)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        except Exception:
            pass
    return []


def _normalize_llm_candidate(row: dict[str, Any], *, version: str) -> Optional[dict[str, Any]]:
    ver = str(row.get("version") or version or "").strip()
    date_text = str(row.get("date") or row.get("releasedAt") or "").strip()
    if not ver or not date_text:
        return None
    dates = _extract_dates(date_text)
    if not dates:
        return None
    conf = str(row.get("confidence") or "low").strip().lower()
    if conf not in {"low", "medium", "high"}:
        conf = "low"
    reason = str(row.get("reason") or row.get("snippet") or "AI 推断，需人工确认").strip()
    return {
        "version": _normalize_version(ver),
        "date": dates[0],
        "sourceUrl": str(row.get("sourceUrl") or ""),
        "sourceTitle": str(row.get("sourceTitle") or "AI 模型推断（应用市场/公开发布信息）"),
        "snippet": reason[:500],
        "confidence": conf,
        "sourceKind": "llm",
    }


def _suggest_versions_via_llm(
    *,
    product_name: str,
    versions: list[str],
    registration_country: str,
    llm_text_fn: LlmTextFn,
) -> dict[str, list[dict[str, Any]]]:
    if not versions:
        return {}
    markets = "、".join(_market_hints_for_country(registration_country)[:5])
    version_lines = "\n".join(f"- {v}" for v in versions)
    country = (registration_country or "").strip() or "（未提供）"
    prompt = (
        "你是医疗器械独立软件（SaMD）版本发布时间助手。"
        "请优先依据「注册国家对应应用市场」的公开发布/更新记录，查找下列版本的上线或更新日期。\n\n"
        f"产品名称：{product_name or '（未提供）'}\n"
        f"注册国家/地区：{country}\n"
        f"优先检索渠道：{markets}\n"
        f"版本列表：\n{version_lines}\n\n"
        "要求：\n"
        "1. 仅输出 JSON 数组，不要 markdown；\n"
        '2. 每项：{"version":"X.Y.Z.B","date":"YYYY-MM-DD","reason":"来源渠道+简短依据","confidence":"low|medium","sourceTitle":"商店或页面名"}；\n'
        "3. 无可靠公开依据则跳过该版本，禁止编造；\n"
        "4. 全部未知则输出 []。\n"
    )
    logger.info(
        "release-date-search llm batch country=%s versions=%s",
        country,
        ",".join(versions),
    )
    try:
        raw = llm_text_fn(prompt)
    except Exception as exc:
        logger.warning("release-date-search llm failed: %s", exc)
        return {}
    rows = _parse_llm_json_array(raw)
    out: dict[str, list[dict[str, Any]]] = {v: [] for v in versions}
    for row in rows:
        ver_raw = str(row.get("version") or "").strip()
        if not ver_raw:
            continue
        try:
            matched = _normalize_version(ver_raw)
        except ValueError:
            continue
        if matched not in out:
            continue
        candidate = _normalize_llm_candidate(row, version=matched)
        if candidate:
            out[matched].append(candidate)
    return out


def _merge_llm_into_per_version(
    per_version: list[dict[str, Any]],
    llm_map: dict[str, list[dict[str, Any]]],
) -> None:
    for block in per_version:
        version = str(block.get("version") or "").strip()
        if not version or (block.get("candidates") or []):
            continue
        llm_rows = llm_map.get(version) or []
        if not llm_rows:
            continue
        block["candidates"] = llm_rows[:3]
        block["message"] = "网页检索无结果，已使用 AI 按注册国家应用市场推断（须人工确认）。"
        if isinstance(block.get("diagnostics"), dict):
            block["diagnostics"]["llmUsed"] = True
            block["diagnostics"]["llmCandidateCount"] = len(llm_rows)


def _itunes_storefronts_for_region(registration_country: str) -> list[str]:
    region = normalize_registration_region(registration_country)
    if _is_china_country(region):
        return ["cn"]
    key = (region or "").casefold()
    if any(x in key for x in ("美国", "usa", "us")):
        return ["us"]
    if any(x in key for x in ("英国", "uk")):
        return ["gb", "us"]
    if any(x in key for x in ("日本", "japan", "jp")):
        return ["jp", "us"]
    if any(x in key for x in ("澳大利亚", "australia", "au")):
        return ["au", "us"]
    if any(x in key for x in ("加拿大", "canada", "ca")):
        return ["ca", "us"]
    if any(x in key for x in ("欧盟", "欧洲", "eu", "ce")):
        # 欧洲区优先荷兰/德国（鱼跃 EU 主体常见），再扩散
        return ["nl", "de", "fr", "gb", "ie", "it", "es", "be", "at"]
    return ["us", "de", "gb", "cn"]


def _normalize_loose_version(raw: str) -> Optional[str]:
    """App Store 版本可能是 1.0.1.7 / 1.0.1 / V2.1.1 → 尽量规范化到 X.Y.Z.B。"""
    text = re.sub(r"^[Vv]", "", (raw or "").strip())
    if not text:
        return None
    if VERSION_RE.match(text):
        return _normalize_version(text)
    m = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:\.(\d+))?$", text)
    if not m:
        return None
    parts = [int(m.group(i) or 0) for i in range(1, 5)]
    return f"{parts[0]}.{parts[1]}.{parts[2]}.{parts[3]}"


def _itunes_name_score(track_name: str, product_name: str) -> int:
    a = re.sub(r"\s+", " ", (track_name or "").strip()).casefold()
    b = re.sub(r"\s+", " ", (product_name or "").strip()).casefold()
    if not a or not b:
        return -100
    if a == b:
        return 100
    if a.startswith(b) or b.startswith(a):
        return 80
    if b in a or a in b:
        return 60
    ta = {t for t in re.split(r"[\s\-_/]+", a) if len(t) >= 2}
    tb = {t for t in re.split(r"[\s\-_/]+", b) if len(t) >= 2}
    if not ta or not tb:
        return -20
    overlap = len(ta & tb)
    if overlap == 0:
        return -40
    return 10 + overlap * 8


def _storefront_tz_name(storefront: str) -> str:
    """App Store 区服展示用的日历日时区（与页面 Version History 标签对齐）。"""
    cc = (storefront or "").strip().lower()
    mapping = {
        "us": "America/Los_Angeles",
        "cn": "Asia/Shanghai",
        "jp": "Asia/Tokyo",
        "gb": "Europe/London",
        "ie": "Europe/Dublin",
        "nl": "Europe/Amsterdam",
        "de": "Europe/Berlin",
        "fr": "Europe/Paris",
        "it": "Europe/Rome",
        "es": "Europe/Madrid",
        "be": "Europe/Brussels",
        "at": "Europe/Vienna",
        "au": "Australia/Sydney",
        "ca": "America/Toronto",
    }
    return mapping.get(cc, "UTC")


def _calendar_date_in_tz(dt: datetime, storefront: str = "") -> Optional[str]:
    """把带时区的发布时间折算成对照用的本地日历日。

    App Store Version History 标签随浏览者时区变化；本系统主要供国内用户对照商店页，
    因此优先按 Asia/Shanghai 取日历日（与国内打开 App Store 所见一致），避免比页面少一天。
    """
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # 对照商店页：国内用户所见日期（非裸 UTC、非仅按区服）
        tz_key = "cn"
        try:
            from zoneinfo import ZoneInfo

            local = dt.astimezone(ZoneInfo(_storefront_tz_name(tz_key)))
        except Exception:
            # Windows 缺 tzdata：用固定 UTC+8
            local = dt.astimezone(timezone(timedelta(hours=8)))
        return _fmt_date(local.date())
    except Exception:
        return _fmt_date(dt.date()) if isinstance(dt, datetime) else None


def _parse_itunes_iso_date(raw: str, *, storefront: str = "") -> Optional[str]:
    text = (raw or "").strip()
    if not text:
        return None
    # 2026-06-26T22:58:11Z / 2026-06-26T22:58:11.000+00:00
    try:
        normalized = text.replace("Z", "+00:00")
        if re.match(r"^20\d{2}-\d{2}-\d{2}T", normalized):
            dt = datetime.fromisoformat(normalized)
            return _calendar_date_in_tz(dt, storefront)
    except Exception:
        pass
    m = re.match(r"^(20\d{2})-(\d{2})-(\d{2})", text)
    if not m:
        return None
    return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _itunes_http_get_json(url: str) -> tuple[Optional[dict[str, Any]], str]:
    """iTunes API 优先直连（公开接口），失败再复用 LLM 代理。"""
    from config.cursor_overrides import build_llm_httpx_client

    headers = {
        "User-Agent": "aicheckword-release-date/1.0 (itunes-search)",
        "Accept": "application/json",
        "Connection": "close",
    }
    last_err = ""
    for label, builder in (
        (
            "itunes_direct",
            lambda: httpx.Client(
                timeout=_llm_shared_httpx_timeout(20.0),
                trust_env=False,
                http2=False,
                follow_redirects=True,
                headers=headers,
            ),
        ),
        (
            "itunes_llm_shared",
            lambda: build_llm_httpx_client(timeout=_llm_shared_httpx_timeout(20.0)),
        ),
    ):
        try:
            with builder() as client:
                resp = client.get(url, headers=headers, follow_redirects=True)
                resp.raise_for_status()
                data = resp.json()
            if isinstance(data, dict):
                return data, label
            last_err = "响应非 JSON 对象"
        except Exception as exc:
            last_err = f"{label}:{exc}"
    return None, last_err


def _parse_app_store_js_date(raw: str, *, storefront: str = "") -> Optional[str]:
    """解析 App Store 页面 secondarySubtitle：Fri Jun 26 2026 22:58:11 GMT+0000 (...)。

    返回区服本地日历日，与商店 Version History 展示（如 27 Jun）对齐。
    """
    text = (raw or "").strip()
    if not text:
        return None
    # 已是 ISO
    iso = _parse_itunes_iso_date(text, storefront=storefront)
    if iso:
        return iso
    # Fri Jun 26 2026 22:58:11 GMT+0000 (Coordinated Universal Time)
    m = re.match(
        r"^([A-Za-z]{3}\s+[A-Za-z]{3}\s+\d{1,2}\s+20\d{2}\s+\d{2}:\d{2}:\d{2})\s*"
        r"GMT([+-])(\d{2})(\d{2})",
        text,
    )
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%a %b %d %Y %H:%M:%S")
            sign = 1 if m.group(2) == "+" else -1
            offset = timezone(
                timedelta(hours=sign * int(m.group(3)), minutes=sign * int(m.group(4)))
            )
            dt = dt.replace(tzinfo=offset)
            return _calendar_date_in_tz(dt, storefront)
        except Exception:
            pass
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    text = re.sub(r"\s*GMT[+-]\d{4}.*$", "", text, flags=re.I).strip()
    for fmt in ("%a %b %d %Y %H:%M:%S", "%a %b %d %Y"):
        try:
            dt = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            return _calendar_date_in_tz(dt, storefront)
        except Exception:
            continue
    return None


_APP_STORE_VERSION_SUBTITLE_RE = re.compile(
    r'"primarySubtitle"\s*:\s*"(?:(?:Versie|Version)\s*)?([^"]+?)"\s*,\s*'
    r'"secondarySubtitle"\s*:\s*"([^"]*?GMT[^"]*)"',
    re.I,
)


def _parse_app_store_page_version_history(
    html: str, *, storefront: str = ""
) -> dict[str, str]:
    """从 apps.apple.com 页面 serialized-server-data 抽取 版本→发布日。"""
    out: dict[str, str] = {}
    blob = html or ""
    m = re.search(
        r'<script[^>]*id="serialized-server-data"[^>]*>(.*?)</script>',
        blob,
        flags=re.S | re.I,
    )
    if m:
        blob = m.group(1)
    for ver_raw, date_raw in _APP_STORE_VERSION_SUBTITLE_RE.findall(blob):
        ver = _normalize_loose_version(ver_raw) or (ver_raw or "").strip()
        released = _parse_app_store_js_date(date_raw, storefront=storefront)
        if not ver or not released:
            continue
        # 保留首次（页面从上到下通常已是新→旧）；不覆盖已有
        if ver not in out:
            out[ver] = released
    return out


def _fetch_app_store_page_html(track_id: Any, storefront: str) -> tuple[str, str]:
    """经 LLM/Cursor 同款代理抓取 App Store 产品页（国内直连会被重定向到 Today）。"""
    from config.cursor_overrides import build_llm_httpx_client

    cc = (storefront or "us").strip().lower() or "us"
    tid = str(track_id or "").strip()
    if not tid:
        return "", "missing trackId"
    urls = [
        f"https://apps.apple.com/{cc}/app/id{tid}?l=en",
        f"https://apps.apple.com/{cc}/app/id{tid}",
    ]
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "close",
    }
    last_err = ""
    for url in urls:
        try:
            with build_llm_httpx_client(timeout=_llm_shared_httpx_timeout(30.0)) as client:
                resp = client.get(url, headers=headers, follow_redirects=True)
                resp.raise_for_status()
                final = str(resp.url)
                html = resp.text or ""
            if "/iphone/today" in final or "serialized-server-data" not in html:
                last_err = f"页面被重定向或无应用数据 final={final[:120]}"
                continue
            return html, url
        except Exception as exc:
            last_err = str(exc)
    return "", last_err or "App Store 产品页抓取失败"


def _resolve_app_store_app(
    *,
    product_name: str,
    registration_country: str,
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    """定位 App Store 应用（命中高分后立即停止扫区，避免每个版本扫遍欧洲）。"""
    product = (product_name or "").strip()
    storefronts = _itunes_storefronts_for_region(registration_country)
    diagnostics: dict[str, Any] = {
        "storefrontsTried": [],
        "rawHits": 0,
        "networkError": None,
    }
    best_app: Optional[dict[str, Any]] = None
    best_score = -10**9
    terms = [product]
    alt = re.sub(r"\s+by\s+\w+$", "", product, flags=re.I).strip()
    if alt and alt.casefold() != product.casefold():
        terms.append(alt)

    for cc in storefronts:
        for term in terms:
            url = (
                f"{_ITUNES_SEARCH_URL}?term={quote(term)}"
                f"&entity=software&country={quote(cc)}&limit=12"
            )
            data, attempt = _itunes_http_get_json(url)
            diagnostics["storefrontsTried"].append(
                {"country": cc, "term": term, "attempt": attempt, "ok": bool(data)}
            )
            if not data:
                diagnostics["networkError"] = attempt
                continue
            rows = data.get("results") if isinstance(data.get("results"), list) else []
            diagnostics["rawHits"] = int(diagnostics["rawHits"] or 0) + len(rows)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                track = str(row.get("trackName") or "")
                score = _itunes_name_score(track, product)
                if score < 50:
                    continue
                payload = {
                    "trackId": row.get("trackId"),
                    "trackName": track,
                    "sellerName": row.get("sellerName") or "",
                    "version": _normalize_loose_version(str(row.get("version") or ""))
                    or str(row.get("version") or ""),
                    "currentVersionReleaseDate": _parse_itunes_iso_date(
                        str(row.get("currentVersionReleaseDate") or ""),
                        storefront=cc,
                    ),
                    "trackViewUrl": row.get("trackViewUrl") or "",
                    "storefront": cc,
                    "nameScore": score,
                    "releaseNotes": str(row.get("releaseNotes") or "")[:400],
                }
                if score > best_score:
                    best_score = score
                    best_app = payload
                # 产品名精确/高相似度：无需继续扫其它国家
                if score >= 80 and payload.get("trackId"):
                    return best_app, diagnostics
        if best_app and best_score >= 80:
            break
    return best_app, diagnostics


def _search_app_store_release(
    *,
    product_name: str,
    version: str,
    registration_country: str,
    app_cache: Optional[dict[str, Any]] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """iTunes 定位应用 + App Store 产品页版本历史（含历史小版本）。"""
    product = (product_name or "").strip()
    target_ver = _normalize_loose_version(version) or (version or "").strip()
    cache_key = f"{normalize_registration_region(registration_country)}::{product.casefold()}"
    diagnostics: dict[str, Any] = {
        "engine": "app_store",
        "storefrontsTried": [],
        "networkError": None,
        "rawHits": 0,
        "matchedTrack": None,
        "historyCount": 0,
    }

    cached = (app_cache or {}).get(cache_key) if app_cache is not None else None
    if isinstance(cached, dict) and cached.get("trackId") and isinstance(cached.get("history"), dict):
        best_app = cached
        history: dict[str, str] = dict(cached.get("history") or {})
        diagnostics["matchedTrack"] = {
            k: best_app.get(k)
            for k in (
                "trackId",
                "trackName",
                "sellerName",
                "version",
                "storefront",
                "trackViewUrl",
                "nameScore",
            )
        }
        diagnostics["historyCount"] = len(history)
        diagnostics["cacheHit"] = True
    else:
        best_app, resolve_diag = _resolve_app_store_app(
            product_name=product, registration_country=registration_country
        )
        diagnostics.update({k: resolve_diag.get(k) for k in ("storefrontsTried", "rawHits", "networkError")})
        diagnostics["matchedTrack"] = best_app
        history = {}
        if not best_app or not best_app.get("trackId"):
            diagnostics["failureHint"] = "App Store 未找到同名应用"
            return [], diagnostics

        html, page_meta = _fetch_app_store_page_html(
            best_app.get("trackId"), str(best_app.get("storefront") or "us")
        )
        diagnostics["pageFetch"] = page_meta
        if not html:
            # 回退：至少可用当前版本
            history = {}
            cur_ver = str(best_app.get("version") or "")
            cur_date = best_app.get("currentVersionReleaseDate")
            if cur_ver and cur_date:
                history[cur_ver] = str(cur_date)
            diagnostics["failureHint"] = (
                f"App Store 产品页抓取失败（{page_meta}）；仅有当前版本信息可用"
            )
        else:
            history = _parse_app_store_page_version_history(
                html, storefront=str(best_app.get("storefront") or "us")
            )
            # 补上 lookup 当前版本，防页面解析遗漏
            cur_ver = str(best_app.get("version") or "")
            cur_date = best_app.get("currentVersionReleaseDate")
            if cur_ver and cur_date and cur_ver not in history:
                history[cur_ver] = str(cur_date)
        diagnostics["historyCount"] = len(history)
        if app_cache is not None:
            app_cache[cache_key] = {
                **best_app,
                "history": history,
                "pageUrl": page_meta if html else "",
            }

    released = history.get(target_ver)
    if released:
        cand = {
            "version": target_ver,
            "date": released,
            "sourceUrl": str((best_app or {}).get("trackViewUrl") or ""),
            "sourceTitle": (
                f"App Store（{(best_app or {}).get('storefront') or '-'}）· "
                f"{(best_app or {}).get('trackName') or product}"
            ),
            "snippet": (
                f"{(best_app or {}).get('sellerName') or ''}; "
                f"version history 匹配 {target_ver}; "
                f"history={len(history)} versions"
            ).strip(),
            "confidence": "high",
            "sourceKind": "app_store",
            "relevanceScore": 140,
        }
        logger.info(
            "release-date-search app_store history hit product=%s version=%s date=%s history=%d",
            product,
            target_ver,
            released,
            len(history),
        )
        return [cand], diagnostics

    if history:
        sample = ", ".join(list(history.keys())[:8])
        diagnostics["failureHint"] = (
            f"已找到 App Store「{(best_app or {}).get('trackName')}」版本历史 "
            f"（{len(history)} 个：{sample}…），但不含目标版本 {target_ver}"
        )
    elif not diagnostics.get("failureHint"):
        diagnostics["failureHint"] = f"App Store 未解析到版本历史（目标 {target_ver}）"
    return [], diagnostics


def _suggest_release_date_for_version(
    *,
    product_name: str,
    version: str,
    registration_country: str = "",
    include_diagnostics: bool = False,
    app_cache: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    product = (product_name or "").strip()
    country = (registration_country or "").strip()
    if not product:
        payload: dict[str, Any] = {
            "version": version,
            "query": "",
            "queries": [],
            "registrationCountry": country,
            "candidates": [],
            "message": "请先填写产品名称后再检索（需按产品名精确匹配，避免搜到其它产品/国内页）。",
        }
        if include_diagnostics:
            payload["diagnostics"] = {
                "duckduckgo": {"engine": "none", "rawHits": 0, "failureHint": "缺少产品名称"},
                "dateExtraction": {"failureHint": "缺少产品名称"},
                "queriesTried": [],
                "marketHints": _market_hints_for_country(country),
            }
        return payload

    used_queries: list[str] = []
    ordered: list[dict[str, Any]] = []
    extraction: dict[str, Any] = {}
    last_diag: dict[str, Any] = {}

    # 1) 优先 App Store：iTunes 定位 + 产品页版本历史（含历史小版本）
    app_candidates, app_diag = _search_app_store_release(
        product_name=product,
        version=version,
        registration_country=country,
        app_cache=app_cache,
    )
    last_diag = {"engine": "app_store", **app_diag}
    used_queries.append(f"App Store API · {product} · {version}")
    if app_candidates:
        ordered = app_candidates
        extraction = {
            "candidateCount": len(ordered),
            "source": "app_store",
            "productName": product,
            "registrationCountry": country,
        }
    else:
        # 2) 网页搜索仅作补充（Bing HTML 常被验证码拦截）
        queries = _build_search_queries(
            product_name=product,
            version=version,
            registration_country=country,
            max_queries=_MAX_QUERIES_PER_VERSION,
        )
        merged_results: list[dict[str, str]] = []
        for qi, query in enumerate(queries):
            if qi > 0:
                time.sleep(0.35)
            results, web_diag = _web_search_with_diagnostics(
                query, registration_country=country
            )
            used_queries.append(query)
            last_diag = web_diag
            last_diag["appStore"] = app_diag
            if results:
                merged_results.extend(results)
                seen_url: set[str] = set()
                uniq_results: list[dict[str, str]] = []
                for row in merged_results:
                    u = row.get("url") or ""
                    if u in seen_url:
                        continue
                    seen_url.add(u)
                    uniq_results.append(row)
                ordered, extraction = _build_candidates_from_results(
                    version=version,
                    results=uniq_results,
                    include_skipped=include_diagnostics,
                    product_name=product,
                    registration_country=country,
                )
                if ordered:
                    break
            if web_diag.get("networkError") and web_diag.get("proxyConfigured"):
                break
        if not ordered and merged_results:
            seen_url = set()
            uniq_results = []
            for row in merged_results:
                u = row.get("url") or ""
                if u in seen_url:
                    continue
                seen_url.add(u)
                uniq_results.append(row)
            ordered, extraction = _build_candidates_from_results(
                version=version,
                results=uniq_results,
                include_skipped=include_diagnostics,
                product_name=product,
                registration_country=country,
            )
        if not ordered and app_diag.get("failureHint"):
            extraction["failureHint"] = app_diag.get("failureHint")

    logger.info(
        "release-date-search version=%s country=%s candidates=%d queries=%d engine=%s",
        version,
        registration_country or "-",
        len(ordered),
        len(used_queries),
        (last_diag or {}).get("engine") or "none",
    )
    country_label = country or "未指定"
    if ordered:
        src = ordered[0].get("sourceKind") or "web"
        prefix = "App Store" if src == "app_store" else "网页"
        message = f"已从{prefix}按产品「{product}」、注册地区「{country_label}」检索到候选日期。"
    elif last_diag.get("networkError") and not app_diag.get("matchedTrack"):
        message = last_diag.get("failureHint") or "应用市场网页检索失败（常需代理）。"
    elif extraction.get("failureHint") or app_diag.get("failureHint"):
        message = str(extraction.get("failureHint") or app_diag.get("failureHint"))
    else:
        message = (
            f"未检索到产品「{product}」在注册地区「{country_label}」的该版本发布时间。"
        )

    payload: dict[str, Any] = {
        "version": version,
        "query": used_queries[0] if used_queries else "",
        "queries": used_queries,
        "registrationCountry": registration_country or "",
        "candidates": ordered,
        "message": message,
    }
    if include_diagnostics:
        payload["diagnostics"] = {
            "webSearch": last_diag,
            "appStore": app_diag,
            "duckduckgo": last_diag,  # 兼容旧诊断字段名
            "dateExtraction": extraction,
            "queriesTried": used_queries,
            "marketHints": _market_hints_for_country(registration_country),
        }
    return payload


def suggest_release_dates(
    *,
    product_name: str,
    from_version: str,
    to_version: str,
    intermediate_versions: Optional[list[str]] = None,
    target_version: Optional[str] = None,
    registration_country: str = "",
    include_diagnostics: bool = False,
    llm_text_fn: Optional[LlmTextFn] = None,
) -> dict[str, Any]:
    versions = _parse_version_chain(from_version, to_version, intermediate_versions)
    targets = versions
    if target_version:
        normalized = _normalize_version(target_version)
        if normalized not in versions:
            raise ValueError(f"目标版本 {target_version} 不在版本链路中")
        targets = [normalized]

    raw_country = (registration_country or "").strip()
    country = normalize_registration_region(raw_country)
    product = (product_name or "").strip()
    logger.info(
        "release-date-search chain=%s targets=%s product=%s country=%s (raw=%s)",
        "->".join(versions),
        targets,
        product or "-",
        country or "-",
        raw_country or "-",
    )
    per_version: list[dict[str, Any]] = []
    needs_llm: list[str] = []
    # 同一产品的 App Store 应用/版本历史只解析一次，供链路各版本复用
    app_cache: dict[str, Any] = {}
    for idx, version in enumerate(targets):
        if idx > 0:
            time.sleep(_DDG_INTER_REQUEST_SLEEP_SEC)
        result = _suggest_release_date_for_version(
            product_name=product,
            version=version,
            registration_country=country or raw_country,
            include_diagnostics=include_diagnostics,
            app_cache=app_cache,
        )
        # 便于前端展示映射关系
        result["registrationCountryRaw"] = raw_country
        result["registrationRegion"] = country or raw_country
        per_version.append(result)
        if not (result.get("candidates") or []):
            needs_llm.append(version)

    llm_used = False
    # 无产品名时禁止 LLM：既浪费时间又易拖垮上游超时，aiword 会误报「文档服务暂不可用」
    if needs_llm and llm_text_fn and product:
        llm_map = _suggest_versions_via_llm(
            product_name=product,
            versions=needs_llm,
            registration_country=country or raw_country,
            llm_text_fn=llm_text_fn,
        )
        if any(llm_map.get(v) for v in needs_llm):
            llm_used = True
            _merge_llm_into_per_version(per_version, llm_map)

    all_candidates: list[dict[str, Any]] = []
    for block in per_version:
        for candidate in block.get("candidates") or []:
            all_candidates.append(candidate)

    versions_missing = [
        str(x.get("version") or "")
        for x in per_version
        if not (x.get("candidates") or [])
    ]
    summary = {
        "versionCount": len(targets),
        "candidateCount": len(all_candidates),
        "versionsWithCandidates": sum(1 for x in per_version if (x.get("candidates") or [])),
        "totalRawHits": sum(
            int((x.get("diagnostics") or {}).get("duckduckgo", {}).get("rawHits") or 0)
            for x in per_version
        ),
        "llmUsed": llm_used,
        "llmFallbackVersions": needs_llm if llm_used else [],
        "registrationCountry": country or raw_country,
        "registrationCountryRaw": raw_country,
        "registrationRegion": country or raw_country,
        "productName": product,
        "marketHints": _market_hints_for_country(country or raw_country),
        "versionsMissing": versions_missing,
    }
    if include_diagnostics or True:
        if not all_candidates:
            proxy_modes = [
                str(((x.get("diagnostics") or {}).get("duckduckgo") or {}).get("proxyMode") or "")
                for x in per_version
            ]
            first_hint = next(
                (
                    str(((x.get("diagnostics") or {}).get("duckduckgo") or {}).get("failureHint") or "")
                    for x in per_version
                    if ((x.get("diagnostics") or {}).get("duckduckgo") or {}).get("failureHint")
                ),
                "",
            )
            if first_hint:
                summary["failureHint"] = first_hint
            elif any(m in {"none", ""} for m in proxy_modes) and not any(
                m == "llm_shared" for m in proxy_modes
            ):
                summary["failureHint"] = (
                    "未检索到发布时间（已尝试复用 Cursor/LLM 代理）。"
                    "请确认侧栏/系统配置里 Cursor 在用的代理对本机仍可用"
                )
            else:
                summary["failureHint"] = (
                    f"未在注册国家「{country or '未指定'}」检索到可解析的发布时间，请手动填写"
                )

    source = "aicheckword"
    if llm_used:
        source = "aicheckword+llm"

    if not all_candidates:
        message = (
            f"未检索到发布时间（注册国家：{country or '未指定'}）。"
            "请检查产品名/版本号，或改用手动填写。"
        )
    elif versions_missing:
        message = (
            f"部分版本已找到候选日期；仍有 {len(versions_missing)} 个版本未找到，请手动填写或采用候选。"
        )
    elif llm_used:
        message = "部分结果来自 AI 按应用市场推断，须人工确认后采用。"
    else:
        message = "已检索到候选发布时间，请人工确认后采用。"

    return {
        "fromVersion": versions[0],
        "toVersion": versions[-1],
        "versionChain": versions,
        "targetVersion": target_version or None,
        "registrationCountry": country,
        "perVersion": per_version,
        "candidates": all_candidates[:20],
        "message": message,
        "source": source,
        "diagnostics": summary if include_diagnostics else {
            "candidateCount": len(all_candidates),
            "versionsMissing": versions_missing,
            "registrationCountry": country,
            "failureHint": summary.get("failureHint") if not all_candidates else None,
        },
    }


def diagnose_release_dates(
    *,
    product_name: str,
    from_version: str,
    to_version: str,
    intermediate_versions: Optional[list[str]] = None,
    target_version: Optional[str] = None,
    registration_country: str = "",
    llm_text_fn: Optional[LlmTextFn] = None,
) -> dict[str, Any]:
    result = suggest_release_dates(
        product_name=product_name,
        from_version=from_version,
        to_version=to_version,
        intermediate_versions=intermediate_versions,
        target_version=target_version,
        registration_country=registration_country,
        include_diagnostics=True,
        llm_text_fn=llm_text_fn,
    )
    result["mode"] = "diagnose"
    return result
