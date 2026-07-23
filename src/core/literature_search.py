"""文献检索（PubMed / Google Scholar）。

出站代理与 Cursor/初稿共用 ``get_llm_http_proxy()``，不另配代理。
供 aiword ``/literature`` 经 integration API 调用。
"""

from __future__ import annotations

import html
import random
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional
from urllib.parse import quote_plus

import httpx

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# 仅主结果卡（gs_or），排除侧栏/相关推荐等纯 gs_r，避免掺条与打乱浏览器顺序
_RESULT_START_RE = re.compile(
    r'<div class="gs_r\b[^"]*\bgs_or\b[^"]*"[^>]*>',
    re.I,
)
# 回退：部分镜像页可能无 gs_or
_RESULT_START_FALLBACK_RE = re.compile(r'<div class="gs_r\b[^"]*"[^>]*>', re.I)
_TITLE_RE = re.compile(r'<h3 class="gs_rt"[^>]*>(.*?)</h3>', re.S | re.I)
_LINK_RE = re.compile(r'<a[^>]+href="([^"]+)"', re.I)
_META_RE = re.compile(r'<div class="gs_a"[^>]*>(.*?)</div>', re.S | re.I)
_DATA_RP_RE = re.compile(r'\bdata-rp=["\']?(\d+)', re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_YEAR_RE = re.compile(r"(19|20)\d{2}")
_SCHOLAR_PAGE_SIZE = 10
# 抓取总时限（秒）：翻页放慢后一次全量抓取更耗时，预算随之抬高；
# 需 ≤ 上游读超时(见 aiword search_service，已同步抬到 1800s)。
_SCHOLAR_FETCH_BUDGET_S = 1500.0
# 翻页间隔（秒）：模拟真人浏览节奏（读完一页再翻），拉长是抗 429 的关键。
# 一次抓 ~20 页大约 4~8 分钟，用「慢而稳」换「一次抓全、少弹验证码」。
_SCHOLAR_PAGE_DELAY_MIN = 12.0
_SCHOLAR_PAGE_DELAY_MAX = 26.0
# 命中 429 后对同一页的最大退避重试次数（更有耐心，别轻易转验证码）
_SCHOLAR_RL_RETRIES = 4
# 续抓时：连续 N 页「零新增」（结果全与已抓重叠）→ 判定该出口 IP 对本检索式
# 已停止提供更多新结果（翻页被 Scholar 折叠/软封锁），停止空转翻页。
_SCHOLAR_OVERLAP_STOP = 2


class ScholarRateLimitedError(RuntimeError):
    """Google Scholar 429 / sorry 人机验证。"""

    def __init__(
        self,
        message: str,
        *,
        captcha_url: str = "",
        cookies: Optional[dict[str, str]] = None,
        html: str = "",
        search_url: str = "",
        partial_records: Optional[list[dict[str, Any]]] = None,
        total_found: int = 0,
        next_offset: int = 0,
    ):
        super().__init__(message)
        self.captcha_url = captcha_url or ""
        self.cookies = dict(cookies or {})
        self.html = html or ""
        self.search_url = search_url or ""
        self.partial_records = list(partial_records or [])
        self.total_found = int(total_found or 0)
        # 原始翻页位置（续抓时从这里接着翻，而非去重后的条数）
        self.next_offset = int(next_offset or 0)


class ScholarFetchError(RuntimeError):
    """Scholar 翻页/代理失败，但可能已抓到部分结果。"""

    def __init__(
        self,
        message: str,
        *,
        partial_records: Optional[list[dict[str, Any]]] = None,
        total_found: int = 0,
        next_offset: int = 0,
    ):
        super().__init__(message)
        self.partial_records = list(partial_records or [])
        self.total_found = int(total_found or 0)
        self.next_offset = int(next_offset or 0)


def _is_proxy_or_ssl_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    tokens = (
        "proxyerror",
        "unable to connect to proxy",
        "ssl",
        "ssleoferror",
        "eof occurred in violation of protocol",
        "certificate verify failed",
        "max retries exceeded",
        "connection reset",
        "connection aborted",
        "10054",
    )
    return any(t in msg for t in tokens)


def _is_scholar_rate_limited(resp: httpx.Response | None = None, exc: BaseException | None = None) -> bool:
    if resp is not None:
        final_url = str(getattr(resp, "url", "") or "").lower()
        if resp.status_code == 429:
            return True
        if "/sorry/" in final_url or "google.com/sorry" in final_url:
            return True
        body = (resp.text or "")[:4000].lower()
        # 勿仅凭单词 captcha（结果页脚本偶含）误判；以 sorry/人机提示为准
        if "/sorry/" in str(getattr(resp, "url", "") or "").lower():
            return True
        if "unusual traffic" in body or "please show you're not a robot" in body:
            return True
        if "our systems have detected unusual traffic" in body:
            return True
    if exc is not None:
        msg = str(exc).lower()
        if "429" in msg or "/sorry/" in msg or "too many requests" in msg:
            return True
    return False


def _scholar_rate_limit_message() -> str:
    return (
        "Google Scholar 触发限流/人机验证（HTTP 429）。"
        "请在弹窗中完成人机验证后点击「验证完成，继续检索」；"
        "或稍后重试 / 改用 RIS/CSV 导入。"
    )


def _raise_rate_limited(
    *,
    resp: httpx.Response | None = None,
    exc: BaseException | None = None,
    search_url: str = "",
    partial_records: Optional[list[dict[str, Any]]] = None,
    total_found: int = 0,
    next_offset: int = 0,
) -> None:
    captcha_url = ""
    cookies: dict[str, str] = {}
    html_text = ""
    if resp is not None:
        captcha_url = str(getattr(resp, "url", "") or "")
        try:
            cookies = {k: v for k, v in resp.cookies.items()}
        except Exception:
            cookies = {}
        html_text = resp.text or ""
    elif isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        captcha_url = str(exc.response.url or "")
        try:
            cookies = {k: v for k, v in exc.response.cookies.items()}
        except Exception:
            cookies = {}
        html_text = exc.response.text or ""
    raise ScholarRateLimitedError(
        _scholar_rate_limit_message(),
        captcha_url=captcha_url or search_url,
        cookies=cookies,
        html=html_text,
        search_url=search_url,
        partial_records=partial_records,
        total_found=total_found,
        next_offset=next_offset,
    )


def _llm_timeout(seconds: float, *, connect: float | None = None) -> httpx.Timeout:
    """分离 connect/read：代理未开时尽快失败，避免每次卡满 read timeout。"""
    read = max(5.0, float(seconds))
    conn = float(connect) if connect is not None else min(8.0, read)
    return httpx.Timeout(read, connect=conn)


def _is_timeout_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "timeout" in msg or "timed out" in msg:
        return True
    return isinstance(exc, (httpx.TimeoutException, TimeoutError))


def _proxy_host_port(proxy_url: str) -> tuple[str, int | None]:
    from urllib.parse import urlparse

    u = urlparse((proxy_url or "").strip())
    host = (u.hostname or "").strip() or "127.0.0.1"
    port = u.port
    return host, port


def _is_transient_outbound_error(exc: BaseException) -> bool:
    """代理抖动、连接重置、超时等可重试错误。"""
    if _is_timeout_error(exc):
        return True
    if _is_proxy_or_ssl_error(exc):
        return True
    msg = str(exc).lower()
    tokens = (
        "connection refused",
        "connection reset",
        "connection aborted",
        "remote disconnected",
        "server disconnected",
        "network is unreachable",
        "temporarily unavailable",
        "proxyerror",
        "tunnel connection failed",
        "10054",
        "10053",
        "10060",
        "read timed out",
        "connect timeout",
        "timed out",
    )
    return any(t in msg for t in tokens)


def _sleep_backoff(attempt: int, *, base: float = 1.2, cap: float = 8.0) -> None:
    """attempt 从 1 开始；带抖动的指数退避。"""
    delay = min(cap, base * (2 ** max(0, attempt - 1))) + random.uniform(0.2, 1.0)
    time.sleep(delay)


def _preflight_llm_proxy(
    proxy_url: str,
    *,
    timeout: float = 2.5,
    retries: int = 3,
) -> Optional[str]:
    """代理端口探测（可重试）；返回错误文案，成功返回 None。"""
    import socket

    host, port = _proxy_host_port(proxy_url)
    if not port:
        return f"llm_http_proxy 缺少端口：{proxy_url}"
    last_exc: Optional[BaseException] = None
    for i in range(max(1, int(retries))):
        try:
            with socket.create_connection((host, int(port)), timeout=timeout):
                return None
        except OSError as exc:
            last_exc = exc
            if i + 1 < retries:
                _sleep_backoff(i + 1, base=0.8, cap=4.0)
    return (
        f"无法连接代理 {host}:{port}（{last_exc}）。"
        "请先打开 Clash/代理软件，并确认 aicheckword 系统配置 llm_http_proxy "
        "与代理端口一致（当前常见为 http://127.0.0.1:7897）。"
    )


def _outbound_get_once(
    url: str,
    *,
    params: Optional[dict[str, Any]],
    headers: dict[str, str],
    timeout_seconds: float,
    force_proxy: bool,
    cookies: Optional[dict[str, str]],
) -> httpx.Response:
    """单次出站（内部多通道），不含外层重试。"""
    from config.cursor_overrides import (
        build_llm_httpx_client,
        get_llm_http_proxy,
        get_llm_verify_ssl,
        llm_httpx_client_kwargs,
    )

    proxy = (get_llm_http_proxy() or "").strip()
    verify_default = bool(get_llm_verify_ssl())
    connect_s = 8.0 if force_proxy else min(10.0, float(timeout_seconds))
    read_s = float(timeout_seconds)
    if force_proxy:
        read_s = min(read_s, 35.0)
    t = _llm_timeout(read_s, connect=connect_s)
    hdrs = headers or {}

    attempt_builders: list[tuple[str, Callable[[], httpx.Client]]] = []
    if force_proxy and proxy:
        attempt_builders.append(
            (
                "llm_forced_proxy",
                lambda: httpx.Client(
                    proxy=proxy,
                    trust_env=False,
                    verify=verify_default,
                    timeout=t,
                    follow_redirects=True,
                    http2=False,
                ),
            )
        )
        if verify_default:
            attempt_builders.append(
                (
                    "llm_forced_proxy_insecure",
                    lambda: httpx.Client(
                        proxy=proxy,
                        trust_env=False,
                        verify=False,
                        timeout=t,
                        follow_redirects=True,
                        http2=False,
                    ),
                )
            )
    elif force_proxy and not proxy:
        raise RuntimeError(
            "Google Scholar 需要代理，但未配置 llm_http_proxy。"
            "请在 aicheckword 系统配置填写显式代理（如 http://127.0.0.1:7897），"
            "或在 .env 设置 HTTP_PROXY。"
        )
    else:
        attempt_builders.append(("llm_shared_mounts", lambda: build_llm_httpx_client(timeout=t)))
        attempt_builders.append(
            (
                "llm_shared_for_url",
                lambda: httpx.Client(
                    **{
                        **llm_httpx_client_kwargs(for_url=url),
                        "timeout": t,
                        "follow_redirects": True,
                    }
                ),
            )
        )
        attempt_builders.append(
            (
                "direct",
                lambda: httpx.Client(
                    trust_env=False,
                    verify=verify_default,
                    timeout=t,
                    follow_redirects=True,
                    http2=False,
                ),
            )
        )
        if verify_default:
            attempt_builders.append(
                (
                    "direct_insecure",
                    lambda: httpx.Client(
                        trust_env=False,
                        verify=False,
                        timeout=t,
                        follow_redirects=True,
                        http2=False,
                    ),
                )
            )

    last_err: Optional[BaseException] = None
    cookie_map = dict(cookies or {})
    timed_out = False
    for _label, builder in attempt_builders:
        try:
            with builder() as client:
                if cookie_map:
                    for k, v in cookie_map.items():
                        try:
                            client.cookies.set(str(k), str(v), domain=".google.com")
                        except Exception:
                            try:
                                client.cookies.set(str(k), str(v))
                            except Exception:
                                pass
                resp = client.get(url, params=params, headers=hdrs)
                if force_proxy:
                    # Scholar 页面统一按 UTF-8 解码（须在首次访问 .text 前设定，
                    # 否则限流检测读过 .text 后再改会报错）
                    try:
                        resp.encoding = "utf-8"
                    except Exception:
                        pass
                if _is_scholar_rate_limited(resp=resp):
                    _raise_rate_limited(resp=resp, search_url=url)
                resp.raise_for_status()
                return resp
        except ScholarRateLimitedError:
            raise
        except Exception as exc:
            last_err = exc
            if _is_timeout_error(exc):
                timed_out = True
            if _is_scholar_rate_limited(exc=exc):
                _raise_rate_limited(exc=exc, search_url=url)
            if isinstance(exc, httpx.HTTPStatusError):
                raise
            continue

    if force_proxy and proxy and timed_out:
        probe = _preflight_llm_proxy(proxy, retries=2)
        extra = f" {probe}" if probe else (
            f" 代理 {proxy} 端口可连，但访问 Scholar 仍然超时；"
            "请在 Clash 开启「系统代理/允许局域网」，并确认规则未拦截 google。"
        )
        raise RuntimeError(
            f"经 llm_http_proxy 访问 Google Scholar 超时。{extra} 原始错误：{last_err}"
        ) from last_err

    hint = (
        "文献外网请求失败（已复用初稿/Cursor 的 llm_http_proxy）。"
        "请在 aicheckword 系统配置确认 llm_http_proxy 或 .env 的 HTTP_PROXY 可用。"
    )
    raise RuntimeError(f"{hint} 原始错误：{last_err}") from last_err


def _outbound_get(
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout_seconds: float = 25,
    force_proxy: bool = False,
    cookies: Optional[dict[str, str]] = None,
    retries: int = 3,
) -> httpx.Response:
    """复用初稿/Cursor 的 llm_http_proxy；对代理抖动自动退避重试。"""
    hdrs = headers or {}
    attempts = max(1, int(retries))
    last_err: Optional[BaseException] = None
    for i in range(attempts):
        try:
            return _outbound_get_once(
                url,
                params=params,
                headers=hdrs,
                timeout_seconds=timeout_seconds,
                force_proxy=force_proxy,
                cookies=cookies,
            )
        except ScholarRateLimitedError:
            raise
        except httpx.HTTPStatusError:
            raise
        except Exception as exc:
            last_err = exc
            # 配置类错误不重试
            msg = str(exc)
            if "未配置 llm_http_proxy" in msg or "缺少端口" in msg:
                raise
            if not _is_transient_outbound_error(exc) and "超时" not in msg and "timed out" not in msg.lower():
                raise
            if i + 1 >= attempts:
                break
            # 重试前再探一次代理端口（可能刚重启 Clash）
            if force_proxy:
                try:
                    from config.cursor_overrides import get_llm_http_proxy

                    proxy_url = (get_llm_http_proxy() or "").strip()
                    if proxy_url:
                        _preflight_llm_proxy(proxy_url, retries=2)
                except Exception:
                    pass
            _sleep_backoff(i + 1, base=1.5, cap=10.0)
    assert last_err is not None
    raise last_err


def _build_pubmed_term(query: str, start_year: int | None, end_year: int | None) -> str:
    q = (query or "").strip()
    if not q:
        return ""
    if start_year and end_year:
        return f"({q}) AND ({start_year}:{end_year}[dp])"
    if start_year:
        return f"({q}) AND ({start_year}:3000[dp])"
    if end_year:
        return f"({q}) AND (1900:{end_year}[dp])"
    return q


def _xml_text(elem: ET.Element | None, path: str) -> str:
    if elem is None:
        return ""
    hit = elem.find(path)
    if hit is None or hit.text is None:
        return ""
    return hit.text.strip()


def _fore_to_initials(fore: str) -> str:
    parts = re.split(r"[\s\-]+", (fore or "").strip())
    return "".join(p[0].upper() for p in parts if p)


def _pubmed_authors(article_elem: ET.Element) -> str:
    """Vancouver 风格：LastName Initials（与 Clinical Literature Search Result 模板一致）。"""
    names: list[str] = []
    for author in article_elem.findall("./AuthorList/Author"):
        collective = _xml_text(author, "./CollectiveName")
        if collective:
            names.append(collective)
            continue
        last = _xml_text(author, "./LastName")
        fore = _xml_text(author, "./ForeName")
        initials = _xml_text(author, "./Initials") or _fore_to_initials(fore)
        if last and initials:
            names.append(f"{last} {initials}")
        elif last:
            names.append(last)
    return ", ".join(names)


def _pubmed_pub_date(article: ET.Element) -> tuple[str, str]:
    """返回 (year, date_display)，如 ('2021', '2021 Feb 12')。"""
    pub = article.find("./MedlineCitation/Article/Journal/JournalIssue/PubDate")
    if pub is None:
        year = _xml_text(article, "./MedlineCitation/DateCompleted/Year")
        return year, year
    year = _xml_text(pub, "./Year")
    month = _xml_text(pub, "./Month")
    day = _xml_text(pub, "./Day")
    medline = _xml_text(pub, "./MedlineDate")
    if not year and medline:
        ym = _YEAR_RE.search(medline)
        year = ym.group(0) if ym else ""
        return year, medline
    parts = [p for p in (year, month, day) if p]
    return year, " ".join(parts)


def _parse_pubmed_articles(xml_text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    records: list[dict[str, Any]] = []
    for article in root.findall("./PubmedArticle"):
        title = _xml_text(article, "./MedlineCitation/Article/ArticleTitle")
        journal = (
            _xml_text(article, "./MedlineCitation/Article/Journal/ISOAbbreviation")
            or _xml_text(article, "./MedlineCitation/MedlineJournalInfo/MedlineTA")
            or _xml_text(article, "./MedlineCitation/Article/Journal/Title")
        )
        pmid = _xml_text(article, "./MedlineCitation/PMID")
        year, pub_date = _pubmed_pub_date(article)
        issue = article.find("./MedlineCitation/Article/Journal/JournalIssue")
        volume = _xml_text(issue, "./Volume") if issue is not None else ""
        number = _xml_text(issue, "./Issue") if issue is not None else ""
        pages = _xml_text(article, "./MedlineCitation/Article/Pagination/MedlinePgn")
        vip = ""
        if volume and number and pages:
            vip = f"{volume}({number}):{pages}"
        elif volume and pages:
            vip = f"{volume}:{pages}"
        else:
            vip = pages or volume or number
        doi = ""
        for node in article.findall("./PubmedData/ArticleIdList/ArticleId"):
            if (node.attrib.get("IdType") or "").lower() == "doi" and node.text:
                doi = node.text.strip()
                break
        article_elem = article.find("./MedlineCitation/Article") or ET.Element("Article")
        records.append(
            {
                "source": "pubmed",
                "title": title,
                "authors": _pubmed_authors(article_elem),
                "year": year,
                "pub_date": pub_date,
                "journal": journal,
                "volume_issue_pages": vip,
                "doi": doi,
                "pmid": pmid,
                "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            }
        )
    return records


def search_pubmed(
    *,
    query: str,
    start_year: int | None = None,
    end_year: int | None = None,
    max_results: int = 50,
    timeout_seconds: int = 45,
) -> tuple[list[dict[str, Any]], int]:
    """返回 (records, total_found)。按 retstart 翻页拉取，避免只拿到首页。"""
    term = _build_pubmed_term(query, start_year, end_year)
    if not term:
        return [], 0
    want = max(1, min(500, int(max_results)))
    page_size = 200  # E-utilities 单次 id 批量建议上限
    ids: list[str] = []
    total_found = 0
    retstart = 0
    while len(ids) < want:
        batch = min(page_size, want - len(ids))
        params: dict[str, Any] = {
            "db": "pubmed",
            "retmode": "json",
            "retmax": batch,
            "retstart": retstart,
            "term": term,
        }
        search_resp = _outbound_get(
            f"{PUBMED_BASE}/esearch.fcgi",
            params=params,
            timeout_seconds=timeout_seconds,
            force_proxy=False,
        )
        payload = search_resp.json()
        result = (payload or {}).get("esearchresult") or {}
        try:
            total_found = int(result.get("count") or total_found or 0)
        except (TypeError, ValueError):
            pass
        id_list = result.get("idlist") or []
        batch_ids = [str(x).strip() for x in id_list if str(x).strip()]
        if not batch_ids:
            break
        ids.extend(batch_ids)
        retstart += len(batch_ids)
        if len(batch_ids) < batch:
            break
        if total_found and retstart >= total_found:
            break

    ids = ids[:want]
    if not ids:
        return [], total_found

    records: list[dict[str, Any]] = []
    for i in range(0, len(ids), page_size):
        chunk = ids[i : i + page_size]
        fetch_resp = _outbound_get(
            f"{PUBMED_BASE}/efetch.fcgi",
            params={"db": "pubmed", "retmode": "xml", "id": ",".join(chunk)},
            timeout_seconds=timeout_seconds,
            force_proxy=False,
        )
        records.extend(_parse_pubmed_articles(fetch_resp.text))
    return records[:want], total_found


def _clean_html_text(value: str) -> str:
    text = _TAG_RE.sub(" ", value or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _iter_scholar_blocks(html_text: str) -> list[str]:
    """按主结果卡起点切块（优先 gs_or），保持 DOM 顺序 = 浏览器顺序。"""
    text = html_text or ""
    starts = [m.start() for m in _RESULT_START_RE.finditer(text)]
    if not starts:
        starts = [m.start() for m in _RESULT_START_FALLBACK_RE.finditer(text)]
    if not starts:
        return []
    blocks: list[str] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        blocks.append(text[start:end])
    return blocks


def _parse_scholar_entries(html_text: str, max_results: int) -> list[dict[str, Any]]:
    """解析单页结果；严格按 DOM 顺序，不再按 data-rp 重排（避免与浏览器错位）。"""
    records: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    blocks = _iter_scholar_blocks(html_text)
    # 回退：有时页面结构变化，至少按标题节点捞（仍保持出现顺序）
    if not blocks:
        for tmatch in _TITLE_RE.finditer(html_text or ""):
            blocks.append(tmatch.group(0))

    for idx, block in enumerate(blocks):
        if len(records) >= max_results:
            break
        tmatch = _TITLE_RE.search(block)
        if not tmatch:
            continue
        title_html = tmatch.group(1)
        lmatch = _LINK_RE.search(title_html) or _LINK_RE.search(block)
        title = _clean_html_text(title_html)
        # 去掉 Scholar 标题前缀如 [PDF] [HTML] [CITATION]
        title = re.sub(r"^\s*(\[[^\]]+\]\s*)+", "", title).strip()
        if not title:
            continue
        title_key = re.sub(r"\W+", "", title.lower())
        if title_key and title_key in seen_titles:
            continue
        if title_key:
            seen_titles.add(title_key)

        meta_match = _META_RE.search(block)
        meta = meta_match.group(1) if meta_match else ""
        plain = _clean_html_text(meta)
        year = ""
        ym = _YEAR_RE.search(plain)
        if ym:
            year = ym.group(0)
        authors = plain.split("-")[0].strip() if "-" in plain else plain
        journal = ""
        parts = [p.strip() for p in plain.split("-") if p.strip()]
        if len(parts) >= 2:
            # gs_a 常见：Authors - Venue, Year - Publisher；去掉末尾年份避免导出重复
            journal = re.sub(r",?\s*(19|20)\d{2}\s*$", "", parts[1]).strip(" ,;")
        rp_match = _DATA_RP_RE.search(block)
        # data-rp 仅作参考位次；展示顺序以 DOM / 翻页 start 为准
        rank = int(rp_match.group(1)) if rp_match else idx
        link = html.unescape(lmatch.group(1)).strip() if lmatch else ""
        if link.startswith("/"):
            link = "https://scholar.google.com" + link
        doi = _extract_doi_from_text(link, block)
        records.append(
            {
                "source": "scholar",
                "title": title,
                "authors": authors,
                "year": year,
                "journal": journal,
                "volume_issue_pages": "",
                "doi": doi,
                "pmid": "",
                "source_url": link,
                "rank": rank,
            }
        )
    return records


CROSSREF_BASE = "https://api.crossref.org/works"
# 覆盖 doi.org、/doi/abs/10.x、/doi/full/10.x、/doi/pdf/10.x 以及裸 DOI
_DOI_IN_TEXT_RE = re.compile(
    r"(10\.\d{4,9}/[^\s\"'<>?#]+)",
    re.I,
)


def _norm_title_key(value: str) -> str:
    return re.sub(r"\W+", "", (value or "").lower())


def _is_truncated_field(value: str | None) -> bool:
    """Scholar 结果卡常把期刊/作者截断为「Journal of Sleep …」。"""
    v = str(value or "")
    if not v.strip():
        return True
    # \ufffd：解码失败的替换字符（如省略号乱码成「The ��」）
    return (
        "…" in v
        or "..." in v
        or "\ufffd" in v
        or v.rstrip().endswith("…")
        or v.rstrip().endswith("...")
    )


def _clean_doi(raw: str) -> str:
    d = (raw or "").strip().rstrip(".;,)/").lower()
    # 去掉末尾常见非 DOI 尾巴（Wiley 链接常带 /abstract、版本号等）
    d = re.sub(r"/(abstract|full|pdf|epdf|meta)$", "", d)
    return d


def _extract_doi_from_text(*parts: str) -> str:
    for p in parts:
        m = _DOI_IN_TEXT_RE.search(p or "")
        if m:
            return _clean_doi(m.group(1))
    return ""


def _crossref_authors(item: dict[str, Any]) -> str:
    """输出**全名**作者列表（名 + 姓，全部作者，不缩写、不截断），
    与期刊文章页/正式引用一致，例如「Atul Malhotra, Maureen E. Crocker, …」。
    Scholar 的 gs_a 行是缩写并省略的（「A Malhotra … …」），必须用 Crossref 全名覆盖。"""
    names: list[str] = []
    for a in item.get("author") or []:
        fam = (a.get("family") or "").strip()
        given = (a.get("given") or "").strip()
        if given and fam:
            names.append(f"{given} {fam}")
        elif fam:
            names.append(fam)
        elif given:
            names.append(given)
        else:
            # 机构作者
            org = (a.get("name") or "").strip()
            if org:
                names.append(org)
    return ", ".join(names)


def _crossref_year(item: dict[str, Any]) -> str:
    for key in ("published-print", "published-online", "issued", "created"):
        dp = ((item.get(key) or {}).get("date-parts") or [])
        if dp and dp[0]:
            try:
                return str(dp[0][0])
            except Exception:
                continue
    return ""


def _crossref_get_json(
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    timeout_seconds: int = 20,
) -> Optional[dict[str, Any]]:
    """请求 Crossref：先直连，失败再走共享代理（部分环境外网必须经代理）。"""
    headers = {
        "Accept": "application/json",
        "User-Agent": "aicheckword-literature/1.0 (metadata enrichment)",
    }
    for force_proxy in (False, True):
        try:
            resp = _outbound_get(
                url,
                params=params,
                headers=headers,
                timeout_seconds=timeout_seconds,
                force_proxy=force_proxy,
                retries=2,
            )
            return resp.json()
        except Exception:
            continue
    return None


def _crossref_lookup_by_doi(
    doi: str,
    *,
    timeout_seconds: int = 20,
) -> Optional[dict[str, Any]]:
    d = (doi or "").strip().lower()
    if not d.startswith("10."):
        return None
    data = _crossref_get_json(
        f"{CROSSREF_BASE}/{quote_plus(d)}",
        timeout_seconds=timeout_seconds,
    )
    msg = (data or {}).get("message")
    return msg if isinstance(msg, dict) else None


def _title_consistent(rec_title: str, item: dict[str, Any]) -> bool:
    """校验 Crossref 命中项与本记录标题是否为同一篇，避免张冠李戴。

    Scholar 标题常被截断，故除完全相等外，也接受「一方是另一方前缀且足够长」。
    """
    rk = _norm_title_key(re.sub(r"\s*[…\.]{1,3}\s*$", "", rec_title or "").strip())
    titles = item.get("title") or []
    ck = _norm_title_key(titles[0] if titles else "")
    if not rk or not ck:
        return False
    if rk == ck:
        return True
    if len(rk) >= 20 and ck.startswith(rk):
        return True
    if len(ck) >= 20 and rk.startswith(ck):
        return True
    return False


def _crossref_lookup(
    title: str,
    *,
    timeout_seconds: int = 20,
) -> Optional[dict[str, Any]]:
    t = (title or "").strip()
    # 去掉 Scholar 截断省略号，避免影响匹配
    t = re.sub(r"\s*[…\.]{1,3}\s*$", "", t).strip()
    if not t:
        return None
    params = {"query.bibliographic": t, "rows": 5}
    data = _crossref_get_json(CROSSREF_BASE, params=params, timeout_seconds=timeout_seconds)
    items = ((data or {}).get("message") or {}).get("items") or []
    # 仅返回标题一致的命中，宁缺毋滥（一致性判断统一走 _title_consistent）
    for it in items:
        if _title_consistent(title, it):
            return it
    return None


def _apply_crossref(rec: dict[str, Any], item: dict[str, Any], *, trusted: bool) -> None:
    """把 Crossref 字段并入记录。

    ``trusted``（按 DOI 命中且标题一致）时才允许覆盖已有的标题/作者（用权威全名/全称）；
    否则（仅标题检索命中）只**填空**，绝不改动已有 title/authors，避免把别的文献的
    作者串写进标题这类错位。
    """
    ctitle = item.get("container-title") or []
    short = item.get("short-container-title") or []
    journal = (ctitle[0] if ctitle else "").strip() or (short[0] if short else "").strip()
    volume = str(item.get("volume") or "").strip()
    issue = str(item.get("issue") or "").strip()
    page = str(item.get("page") or item.get("article-number") or "").strip()
    doi = str(item.get("DOI") or "").strip().lower()
    year = _crossref_year(item)
    authors = _crossref_authors(item)
    publisher = str(item.get("publisher") or "").strip()
    titles = item.get("title") or []
    full_title = (titles[0] if titles else "").strip()

    vip = ""
    if volume and issue and page:
        vip = f"{volume}({issue}):{page}"
    elif volume and page:
        vip = f"{volume}:{page}"
    else:
        vip = page or volume

    # 期刊：为空时填入；仅在可信匹配时才覆盖被截断的旧值
    cur_journal = (rec.get("journal") or "").strip()
    if not cur_journal:
        rec["journal"] = journal or publisher
    elif trusted and journal and _is_truncated_field(cur_journal):
        rec["journal"] = journal

    # 卷/期/页：仅在为空时填入
    if vip and not (rec.get("volume_issue_pages") or "").strip():
        rec["volume_issue_pages"] = vip

    # 年份：仅在为空时填入
    if year and not (rec.get("year") or "").strip():
        rec["year"] = year
    if year and not (rec.get("pub_date") or "").strip():
        rec["pub_date"] = year

    if trusted:
        # 按 DOI 可信匹配：可用权威 DOI、全名作者，并修正被截断的标题
        if doi:
            rec["doi"] = doi
        if authors:
            rec["authors"] = authors
        if full_title and _is_truncated_field(rec.get("title")):
            rec["title"] = full_title
    else:
        # 仅标题检索命中：只补空，绝不覆盖已有 title/authors
        if doi and not (rec.get("doi") or "").strip():
            rec["doi"] = doi
        if authors and not (rec.get("authors") or "").strip():
            rec["authors"] = authors


def _enrich_one_record(rec: dict[str, Any]) -> None:
    """单条补全（供线程池调用）。

    只在能确认是同一篇文献时才补全：
    - 记录自带 DOI，或 URL 里提取到 DOI，经 Crossref 反查且**标题一致** → 可信(trusted)；
    - 否则按标题检索，命中项**标题一致**才用，且仅填空、不覆盖既有 title/authors。
    """
    rec_title = (rec.get("title") or "").strip()
    existing_doi = (rec.get("doi") or "").strip()
    url_doi = existing_doi or _extract_doi_from_text(
        rec.get("source_url") or "", rec.get("journal") or ""
    )

    is_scholar = (rec.get("source") or "") == "scholar"
    need = (
        is_scholar
        or _is_truncated_field(rec.get("journal"))
        or _is_truncated_field(rec.get("authors"))
        or _is_truncated_field(rec.get("title"))
        or not (rec.get("volume_issue_pages") or "").strip()
        or not existing_doi
    )
    if not need:
        return

    item = None
    trusted = False
    # 1) 按 DOI 反查（记录自带或 URL 提取），必须标题一致才采信，杜绝截断/错误 DOI 张冠李戴
    if url_doi:
        cand = _crossref_lookup_by_doi(url_doi, timeout_seconds=8)
        if cand is not None and (not rec_title or _title_consistent(rec_title, cand)):
            item = cand
            trusted = True
            if not existing_doi:
                rec["doi"] = url_doi
    # 2) DOI 无果时按标题检索（_crossref_lookup 内部已做标题一致校验）
    if item is None:
        cand = _crossref_lookup(rec_title, timeout_seconds=8)
        if cand is not None:
            item = cand
            trusted = False
    if item:
        _apply_crossref(rec, item, trusted=trusted)


def _enrich_records_crossref(
    records: list[dict[str, Any]],
    *,
    max_seconds: float = 90.0,
    max_workers: int = 8,
) -> None:
    """并行补全，并设总时限，避免拖垮整次检索（aiword 读超时）。

    旧实现逐条串行 + sleep，约 20 条就可能超过 6 分钟导致
    ``Read timed out. (read timeout=360)``。
    """
    if not records:
        return

    todo = [
        rec
        for rec in records
        if (
            (rec.get("source") or "") == "scholar"
            or _is_truncated_field(rec.get("journal"))
            or _is_truncated_field(rec.get("authors"))
            or _is_truncated_field(rec.get("title"))
            or not (rec.get("volume_issue_pages") or "").strip()
            or not (rec.get("doi") or "").strip()
        )
    ]
    if not todo:
        return

    workers = max(1, min(int(max_workers), len(todo), 8))
    deadline = time.perf_counter() + max(15.0, float(max_seconds))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_enrich_one_record, rec): rec for rec in todo}
        try:
            for fut in as_completed(futs, timeout=max(1.0, deadline - time.perf_counter())):
                try:
                    fut.result()
                except Exception:
                    pass
                if time.perf_counter() >= deadline:
                    break
        except TimeoutError:
            # 超时后取消未完成任务，保留已补全部分
            for fut in futs:
                fut.cancel()
            return
        # 若已到截止仍有未完成，取消剩余
        if time.perf_counter() >= deadline:
            for fut in futs:
                if not fut.done():
                    fut.cancel()


def _build_scholar_url(
    *,
    query: str,
    start_year: int | None,
    end_year: int | None,
    start: int,
    page_size: int,
    sort_by: str = "relevance",
    hl: str = "zh-CN",
) -> str:
    # 默认 hl=zh-CN，与浏览器结果页排序/命中集一致（用户提供的链接参数）
    lang = (hl or "zh-CN").strip() or "zh-CN"
    # 逐字对齐浏览器 URL：hl=zh-CN & as_sdt=0,5 & 默认 filter（不加 filter=0），
    # 否则「展开类似结果」会改变命中集与排序，导致与浏览器顺序不一致。
    url = (
        "https://scholar.google.com/scholar?"
        f"q={quote_plus(query)}&hl={quote_plus(lang)}&as_sdt=0,5"
        f"&num={page_size}&start={max(0, int(start))}"
    )
    if start_year:
        url += f"&as_ylo={int(start_year)}"
    if end_year:
        url += f"&as_yhi={int(end_year)}"
    if (sort_by or "").strip().lower() in ("date", "datedesc", "scisbd"):
        # Scholar：按日期排序
        url += "&scisbd=1"
    return url


_SCHOLAR_TOTAL_RE = re.compile(
    r"(?:About\s+)?([\d,]+)\s+results|"
    r"(?:找到约|约)\s*([\d,]+)\s*条",
    re.I,
)
# 优先从结果统计条解析（避免正文里「约 N 条」误匹配导致 total 虚低、翻页提前停）
_SCHOLAR_AB_TOTAL_RE = re.compile(
    r'class="[^"]*gs_ab_mdw[^"]*"[^>]*>\s*(?:About\s+)?([\d,]+)\s+results|'
    r'class="[^"]*gs_ab_mdw[^"]*"[^>]*>[^<]*(?:找到约|约)\s*([\d,]+)\s*条|'
    r'id="gs_ab"[^>]*>[\s\S]{0,400}?(?:About\s+)?([\d,]+)\s+results|'
    r'id="gs_ab"[^>]*>[\s\S]{0,400}?(?:找到约|约)\s*([\d,]+)\s*条',
    re.I,
)
_SCHOLAR_NEXT_RE = re.compile(
    # 兼容中英文与按钮/链接两种形态：aria-label=Next/下一页、下一页文案、
    # 「下一页」箭头图标类 gs_ico_nav_next、以及 onclick 里带 start= 的按钮。
    r'aria-label="(?:Next|下一页)"|'
    r'<button[^>]+aria-label="[^"]*"[^>]*>\s*(?:下一页|Next)|'
    r'gs_ico_nav_next|'
    r'onclick="[^"]*[?&]start=\d+|'
    r'<td[^>]*class="[^"]*gs_n[^"]*"[^>]*>.*?(?:Next|下一页)',
    re.I | re.S,
)


def _parse_scholar_total_found(html_text: str) -> int:
    text = html_text or ""
    m = _SCHOLAR_AB_TOTAL_RE.search(text) or _SCHOLAR_TOTAL_RE.search(text)
    if not m:
        return 0
    raw = next((g for g in m.groups() if g), "") or ""
    try:
        return int(str(raw).replace(",", ""))
    except (TypeError, ValueError):
        return 0


def _scholar_has_next_page(html_text: str) -> bool:
    text = html_text or ""
    if _SCHOLAR_NEXT_RE.search(text):
        return True
    # 兼容旧分页：存在 start= 下一页链接
    return bool(re.search(r'href="[^"]*start=\d+[^"]*"', text, re.I))


def _scholar_fetch_page(
    *,
    query: str,
    start_year: int | None,
    end_year: int | None,
    start: int,
    sort_by: str,
    headers: dict[str, str],
    cookie_map: dict[str, str],
    timeout_seconds: int,
    hl: str = "zh-CN",
) -> tuple[str, httpx.Response]:
    url = _build_scholar_url(
        query=query,
        start_year=start_year,
        end_year=end_year,
        start=start,
        page_size=_SCHOLAR_PAGE_SIZE,
        sort_by=sort_by,
        hl=hl,
    )
    resp = _outbound_get(
        url,
        headers=headers,
        timeout_seconds=timeout_seconds,
        force_proxy=True,
        cookies=cookie_map,
        retries=3,
    )
    try:
        cookie_map.update({k: v for k, v in resp.cookies.items()})
    except Exception:
        pass
    return url, resp


def _scholar_warmup_session(
    *,
    headers: dict[str, str],
    cookie_map: dict[str, str],
    timeout_seconds: int,
    hl: str = "zh-CN",
) -> None:
    """访问 Scholar 首页热身，取得 NID/GSP 等 cookie，降低直接翻页触发 429。

    真人是先打开首页再翻页；直接对结果页连点会被判为机器人。失败不抛出，
    仅记录 cookie（拿不到 cookie 也不影响后续，只是更容易被限流）。
    """
    lang = (hl or "zh-CN").strip() or "zh-CN"
    # 仅访问首页一次拿 cookie；不再连点 settings，减少「异常流量」嫌疑
    for warm_url in (f"https://scholar.google.com/?hl={quote_plus(lang)}",):
        try:
            resp = _outbound_get(
                warm_url,
                headers=headers,
                timeout_seconds=timeout_seconds,
                force_proxy=True,
                cookies=cookie_map,
                retries=1,
            )
            try:
                cookie_map.update({k: v for k, v in resp.cookies.items()})
            except Exception:
                pass
            # 首页停留久一点，像真人先看首页再开始检索
            time.sleep(3.5 + random.uniform(1.5, 4.0))
        except ScholarRateLimitedError:
            # 首页就被限流：继续尝试，主循环会处理
            break
        except Exception:
            continue


def search_scholar(
    *,
    query: str,
    start_year: int | None = None,
    end_year: int | None = None,
    max_results: int = 30,
    timeout_seconds: int = 45,
    captcha_session_id: str = "",
    sort_by: str = "relevance",
    start_offset: int = 0,
    enrich: bool = True,
    hl: str = "zh-CN",
    prior_keys: set[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """返回 (records, total_found)。

    策略（保证数据全面、排序对齐浏览器）：
    - 默认 ``hl=zh-CN`` + ``as_sdt=0,5``，与浏览器结果页参数一致；
    - **逐页顺序**抓取（每页 10 条），按 DOM 顺序保留浏览器排序（仅解析 gs_or 主结果）；
    - **不以**顶部「约 N 条」估计值作为翻页硬上限（估计值常虚低，会导致 36/34 提前停）；
    - 每页之间**暂停数秒**降低限流；空页/无下一页才停；
    - 字段截断时用 Crossref（优先 DOI）覆盖补全。
    """
    q = (query or "").strip()
    if not q:
        return [], 0, int(start_offset or 0)
    want = max(1, min(500, int(max_results)))
    page_size = _SCHOLAR_PAGE_SIZE
    offset = max(0, int(start_offset or 0))
    lang = (hl or "zh-CN").strip() or "zh-CN"
    accept_lang = "zh-CN,zh;q=0.9,en;q=0.8" if lang.lower().startswith("zh") else "en-US,en;q=0.9"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept-Language": accept_lang,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": f"https://scholar.google.com/",
    }

    cookie_map: dict[str, str] = {}
    sid = (captcha_session_id or "").strip()
    if sid:
        from src.core.scholar_captcha import get_scholar_captcha_session

        sess = get_scholar_captcha_session(sid)
        if sess:
            cookie_map = dict(sess.get("cookies") or {})

    # 检索前探测代理，避免翻页过程中反复卡满超时
    from config.cursor_overrides import get_llm_http_proxy

    proxy_url = (get_llm_http_proxy() or "").strip()
    if not proxy_url:
        raise RuntimeError(
            "Google Scholar 需要代理，但未配置 llm_http_proxy。"
            "请在 aicheckword 系统配置填写（如 http://127.0.0.1:7897）。"
        )
    probe_err = _preflight_llm_proxy(proxy_url, retries=4)
    if probe_err:
        # 代理刚启动时端口可能间歇失败，再等一轮
        _sleep_backoff(1, base=2.0, cap=5.0)
        probe_err = _preflight_llm_proxy(proxy_url, retries=3)
        if probe_err:
            raise RuntimeError(probe_err)

    # 无既有验证会话时先热身首页拿 cookie，显著降低翻页 429
    if not cookie_map:
        try:
            _scholar_warmup_session(
                headers=headers,
                cookie_map=cookie_map,
                timeout_seconds=timeout_seconds,
                hl=lang,
            )
        except Exception:
            pass

    fetch_deadline = time.monotonic() + _SCHOLAR_FETCH_BUDGET_S

    all_records: list[dict[str, Any]] = []
    # 续抓：用「已抓记录的 key」预置 seen，使本次翻页跳过与已抓重叠的结果，
    # 只把「真正新增」的记录计入，并据此判断该 IP 是否还在提供新结果。
    seen: set[str] = set(k for k in (prior_keys or set()) if k)
    total_found = 0
    first_url = ""
    empty_pages = 0
    overlap_streak = 0
    hard_page_cap = 120
    page_no = 0
    start = offset
    # 「原始翻页位置」：续抓必须用它（而非去重后的条数）作为下次 start，
    # 否则去重会让 offset 落回已抓过的页 → 全被去重 → 误判「没有更多」。
    next_start = offset
    last_html = ""

    while len(all_records) < want and page_no < hard_page_cap:
        # 抓取总时限：超时则优雅停止，返回已抓部分（可点继续检索续抓）
        if time.monotonic() > fetch_deadline:
            break
        if page_no == 0:
            # 首页进入前先像真人一样停顿看一眼（热身后不要立刻连点）
            time.sleep(2.5 + random.uniform(1.0, 3.0))
        else:
            # 翻页间隔模拟真人「读完这页再翻下一页」，抬高下限是抗 429 关键；
            # 每隔几页再加一段更长的「阅读/走神」停顿，进一步降低机器人特征。
            delay = random.uniform(_SCHOLAR_PAGE_DELAY_MIN, _SCHOLAR_PAGE_DELAY_MAX)
            if page_no % 4 == 0:
                delay += random.uniform(15.0, 35.0)
            time.sleep(delay)
        page_no += 1

        transient_retry = 0
        rl_retry = 0
        page_records: list[dict[str, Any]] = []
        budget_stop = False
        while True:
            if time.monotonic() > fetch_deadline:
                budget_stop = True
                break
            rate_limited = False
            resp = None
            url = ""
            captured_rl: Optional[ScholarRateLimitedError] = None
            try:
                url, resp = _scholar_fetch_page(
                    query=q,
                    start_year=start_year,
                    end_year=end_year,
                    start=start,
                    sort_by=sort_by,
                    headers=headers,
                    cookie_map=cookie_map,
                    timeout_seconds=timeout_seconds,
                    hl=lang,
                )
            except ScholarRateLimitedError as rl_exc:
                # 保留原始验证码页信息（captcha_url/cookies/html），弹窗才有内容
                rate_limited = True
                captured_rl = rl_exc
            except Exception as exc:
                # 代理抖动：本页失败可退避重试；已有结果时尽量保留
                if transient_retry < 3 and _is_transient_outbound_error(exc):
                    transient_retry += 1
                    _sleep_backoff(transient_retry, base=2.0, cap=12.0)
                    continue
                if all_records:
                    raise ScholarFetchError(
                        f"Scholar 翻页中断（代理不稳定，已保留前 {len(all_records)} 条；"
                        f"可稍后点继续检索）：{exc}",
                        partial_records=all_records,
                        total_found=total_found,
                        next_offset=start,
                    ) from exc
                raise ScholarFetchError(
                    f"Scholar 请求失败（已重试）：{exc}",
                    partial_records=[],
                    total_found=total_found,
                    next_offset=start,
                ) from exc

            html_text = ""
            if not rate_limited and resp is not None:
                if not first_url:
                    first_url = url
                # 编码已在出站处按 UTF-8 设定
                html_text = resp.text or ""
                last_html = html_text
                parsed_total = _parse_scholar_total_found(html_text)
                # 估计值会波动：取较大值，且绝不把它当翻页硬上限
                if parsed_total > total_found:
                    total_found = parsed_total
                body_l = html_text.lower()
                robot = (
                    "please show you're not a robot" in body_l
                    or "unusual traffic" in body_l
                )
                if _is_scholar_rate_limited(resp=resp) or robot:
                    rate_limited = True

            if rate_limited:
                # 区分两种限流场景：
                # 1) 初始硬封锁（首页就被拦、一条都没抓到）：此时该 IP 已被标记，
                #    多等几分钟也不会解封，反而让用户以为「弹不出验证码」。快速探一次
                #    即转人机验证弹窗，让用户尽快解验证码或换节点。
                # 2) 翻页途中 429（会话本来在工作，只是被临时限速）：值得像真人一样
                #    长退避重试，避免卡在 ~44 条。
                initial_block = (not all_records) and (start == offset)
                max_rl = 1 if initial_block else _SCHOLAR_RL_RETRIES
                if rl_retry < max_rl and time.monotonic() < fetch_deadline:
                    rl_retry += 1
                    if initial_block:
                        # 只快速探一次，别让用户干等
                        wait = random.uniform(6.0, 12.0)
                    else:
                        # 遇限流像真人一样等更久再试（渐进退避，最长约 2.5 分钟）
                        wait = min(150.0, 25.0 * rl_retry + random.uniform(8.0, 20.0))
                    time.sleep(wait)
                    # 翻页途中退避到最后一次前重新热身 cookie（初始硬封锁跳过）
                    if (not initial_block) and rl_retry == _SCHOLAR_RL_RETRIES - 1:
                        try:
                            _scholar_warmup_session(
                                headers=headers,
                                cookie_map=cookie_map,
                                timeout_seconds=timeout_seconds,
                                hl=lang,
                            )
                        except Exception:
                            pass
                    continue
                # 多次退避仍被限流：保留已抓，转人机验证流程（前端可继续）
                if enrich and all_records:
                    try:
                        _enrich_records_crossref(
                            all_records,
                            max_seconds=min(60.0, 8.0 + 0.6 * len(all_records)),
                            max_workers=8,
                        )
                    except Exception:
                        pass
                if captured_rl is not None:
                    # 复用原始验证码页信息（captcha_url/cookies/html），并补上已抓部分
                    captured_rl.partial_records = list(all_records)
                    captured_rl.total_found = total_found
                    # 续抓从「当前被拦的原始页位置」接上，而非去重后的条数
                    captured_rl.next_offset = start
                    if not captured_rl.search_url:
                        captured_rl.search_url = first_url
                    raise captured_rl
                _raise_rate_limited(
                    resp=resp,
                    search_url=first_url or url,
                    partial_records=all_records,
                    total_found=total_found,
                    next_offset=start,
                )

            page_records = _parse_scholar_entries(html_text, max_results=page_size * 3)
            if page_records:
                break
            if page_no == 1 and offset == 0:
                if html_text.strip() and "gs_rt" not in html_text and "gs_r" not in html_text:
                    raise RuntimeError(
                        "Google Scholar 页面未能解析到结果（可能被拦截或页面结构变化）。"
                        "请改用导入或稍后重试。"
                    )
            if transient_retry < 3:
                transient_retry += 1
                _sleep_backoff(transient_retry, base=2.0, cap=10.0)
                continue
            break

        if budget_stop:
            break

        if not page_records:
            empty_pages += 1
            # 连续空页且没有「下一页」→ 结束（不再用虚低的 total_found 硬停）
            if empty_pages >= 2 or not _scholar_has_next_page(last_html):
                break
            # 空页不推进 next_start：续抓仍从「上次真正有新增处」接上，避免偏移空转
            start += page_size
            continue
        empty_pages = 0

        gained = 0
        for rec in page_records:
            key = re.sub(r"\W+", "", (rec.get("title") or "").lower())
            url_key = (rec.get("source_url") or "").strip().lower()
            if url_key and url_key in seen:
                continue
            if key and key in seen:
                continue
            if url_key:
                seen.add(url_key)
            if key:
                seen.add(key)
            # 绝对位次对齐浏览器翻页序号（续抓时从 start_offset 接上）
            rec["rank"] = offset + len(all_records)
            all_records.append(rec)
            gained += 1
            if len(all_records) >= want:
                break
        start += page_size
        if gained > 0:
            # 有真正新增：推进「下一原始页位置」，续抓从这里接上
            next_start = start
            overlap_streak = 0
        else:
            # 本页结果全部与已抓重叠：不推进 next_start（否则续抓偏移会空转到无结果区，
            # 表现为「序号一直加、记录不增加」）。连续多页零新增 → 该出口 IP 已不再
            # 提供更多新结果（翻页被折叠/软封锁），停止翻页。
            overlap_streak += 1
            if overlap_streak >= _SCHOLAR_OVERLAP_STOP:
                break
        # 仅在「取满目标」或「无下一页」时停止；Scholar 顶部「约 N 条」是估计值，不能当硬上限
        if len(all_records) >= want:
            break
        if not _scholar_has_next_page(last_html) and len(page_records) < page_size:
            break

    result = all_records[:want]
    if enrich and result:
        try:
            # 补全总时限随条数放大，但封顶，防止整请求超过上游读超时
            budget = min(120.0, 12.0 + 0.8 * len(result))
            _enrich_records_crossref(result, max_seconds=budget, max_workers=8)
        except Exception:
            pass
    # 补全后仍残留的解码替换字符统一清成省略号，避免展示乱码
    for rec in result:
        for key in ("journal", "authors", "title", "volume_issue_pages"):
            val = rec.get(key)
            if val and "\ufffd" in str(val):
                rec[key] = re.sub(r"\ufffd+", "…", str(val)).strip()
    # 展示用总数：至少不低于已抓条数，避免出现「36/34」这种倒挂
    if len(result) > total_found:
        total_found = len(result)
    return result, total_found, next_start


def run_literature_search(
    *,
    query: str,
    sources: list[str],
    start_year: int | None = None,
    end_year: int | None = None,
    max_results_per_source: int = 30,
    scholar_captcha_session_id: str = "",
    scholar_sort_by: str = "relevance",
    scholar_start_offset: int = 0,
    scholar_hl: str = "zh-CN",
    scholar_prior_keys: list[str] | None = None,
) -> dict[str, Any]:
    from config.cursor_overrides import get_llm_http_proxy
    from src.core.scholar_captcha import create_scholar_captcha_session

    srcs = [str(s or "").strip().lower() for s in (sources or []) if str(s or "").strip()]
    srcs = [s for s in srcs if s in ("pubmed", "scholar")]
    if not srcs:
        raise ValueError("sources 至少包含 pubmed 或 scholar 之一")

    jobs: dict[str, Callable[[], tuple[list[dict[str, Any]], int, int]]] = {}
    if "pubmed" in srcs:
        jobs["pubmed"] = lambda: (
            *search_pubmed(
                query=query,
                start_year=start_year,
                end_year=end_year,
                max_results=max_results_per_source,
            ),
            0,  # pubmed 不用翻页续抓，nextOffset 恒为 0
        )
    if "scholar" in srcs:
        _prior_keyset = set(str(k).strip() for k in (scholar_prior_keys or []) if str(k).strip())
        jobs["scholar"] = lambda: search_scholar(
            query=query,
            start_year=start_year,
            end_year=end_year,
            max_results=max_results_per_source,
            captcha_session_id=scholar_captcha_session_id,
            sort_by=scholar_sort_by,
            start_offset=scholar_start_offset,
            hl=(scholar_hl or "zh-CN").strip() or "zh-CN",
            prior_keys=_prior_keyset,
        )

    details: list[dict[str, Any]] = []
    aggregated: list[dict[str, Any]] = []
    captcha_session_id = ""
    captcha_search_url = ""
    needs_captcha = False
    with ThreadPoolExecutor(max_workers=min(2, len(jobs))) as pool:
        futs = {pool.submit(fn): source for source, fn in jobs.items()}
        for fut in as_completed(futs):
            source = futs[fut]
            t0 = time.perf_counter()
            total_found = 0
            next_offset = 0
            try:
                records, total_found, next_offset = fut.result()
                err = ""
            except ScholarRateLimitedError as exc:
                records = list(getattr(exc, "partial_records", None) or [])
                total_found = int(getattr(exc, "total_found", 0) or 0)
                next_offset = int(getattr(exc, "next_offset", 0) or 0)
                err = str(exc)
                if records:
                    err = f"{err}（已保留本源前 {len(records)} 条；验证后可继续补全翻页）"
                needs_captcha = True
                # 真实 Google 验证页地址（供前端在用户浏览器原生打开解验证，同一出口 IP 放行）
                captcha_search_url = (exc.search_url or exc.captcha_url or "").strip()
                try:
                    captcha_session_id = create_scholar_captcha_session(
                        captcha_url=exc.captcha_url,
                        cookies=exc.cookies,
                        html=exc.html,
                        search_url=exc.search_url,
                    )
                except Exception:
                    captcha_session_id = ""
            except ScholarFetchError as exc:
                records = list(getattr(exc, "partial_records", None) or [])
                total_found = int(getattr(exc, "total_found", 0) or 0)
                next_offset = int(getattr(exc, "next_offset", 0) or 0)
                err = str(exc)
                if records:
                    try:
                        _enrich_records_crossref(
                            records,
                            max_seconds=min(45.0, 6.0 + 0.5 * len(records)),
                            max_workers=6,
                        )
                    except Exception:
                        pass
            except Exception as exc:
                records = []
                err = str(exc)
            details.append(
                {
                    "source": source,
                    "records": records,
                    "error": err,
                    "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                    "totalFound": total_found,
                    "fetched": len(records),
                    "nextOffset": int(next_offset or 0),
                    "needsCaptcha": bool(source == "scholar" and needs_captcha and captcha_session_id),
                    "captchaSessionId": captcha_session_id if source == "scholar" else "",
                }
            )
            aggregated.extend(records)

    proxy = (get_llm_http_proxy() or "").strip()
    return {
        "records": aggregated,
        "details": sorted(details, key=lambda x: x.get("source") or ""),
        "count": len(aggregated),
        "proxyConfigured": bool(proxy),
        "proxyMode": "llm_shared",
        "needsCaptcha": bool(needs_captcha and captcha_session_id),
        "captchaSessionId": captcha_session_id,
        "captchaSource": "scholar" if captcha_session_id else "",
        "captchaSearchUrl": captcha_search_url,
    }
