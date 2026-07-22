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

_RESULT_START_RE = re.compile(r'<div class="gs_r\b[^"]*"[^>]*>', re.I)
_TITLE_RE = re.compile(r'<h3 class="gs_rt"[^>]*>(.*?)</h3>', re.S | re.I)
_LINK_RE = re.compile(r'<a[^>]+href="([^"]+)"', re.I)
_META_RE = re.compile(r'<div class="gs_a"[^>]*>(.*?)</div>', re.S | re.I)
_DATA_RP_RE = re.compile(r'\bdata-rp=["\']?(\d+)', re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_YEAR_RE = re.compile(r"(19|20)\d{2}")
_SCHOLAR_PAGE_SIZE = 10
# 抓取总时限（秒）：留足余量给补全，且不超过上游读超时(900s)
_SCHOLAR_FETCH_BUDGET_S = 600.0
# 翻页间隔（秒）：模拟真人，降低 429；下限抬高是抗限流关键
_SCHOLAR_PAGE_DELAY_MIN = 5.0
_SCHOLAR_PAGE_DELAY_MAX = 11.0
# 命中 429 后对同一页的最大退避重试次数
_SCHOLAR_RL_RETRIES = 3


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
    ):
        super().__init__(message)
        self.captcha_url = captcha_url or ""
        self.cookies = dict(cookies or {})
        self.html = html or ""
        self.search_url = search_url or ""
        self.partial_records = list(partial_records or [])
        self.total_found = int(total_found or 0)


class ScholarFetchError(RuntimeError):
    """Scholar 翻页/代理失败，但可能已抓到部分结果。"""

    def __init__(
        self,
        message: str,
        *,
        partial_records: Optional[list[dict[str, Any]]] = None,
        total_found: int = 0,
    ):
        super().__init__(message)
        self.partial_records = list(partial_records or [])
        self.total_found = int(total_found or 0)


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
    """按结果卡起点切块，避免嵌套 div 导致正则提前截断（旧实现常只解析出少数条）。"""
    text = html_text or ""
    starts = [m.start() for m in _RESULT_START_RE.finditer(text)]
    if not starts:
        return []
    blocks: list[str] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        blocks.append(text[start:end])
    return blocks


def _parse_scholar_entries(html_text: str, max_results: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    blocks = _iter_scholar_blocks(html_text)
    # 回退：有时页面结构变化，至少按标题节点捞
    if not blocks:
        for tmatch in _TITLE_RE.finditer(html_text or ""):
            blocks.append(tmatch.group(0))

    ranked: list[tuple[int, dict[str, Any]]] = []
    for idx, block in enumerate(blocks):
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
        rank = int(rp_match.group(1)) if rp_match else idx
        link = html.unescape(lmatch.group(1)).strip() if lmatch else ""
        if link.startswith("/"):
            link = "https://scholar.google.com" + link
        # 侧栏 PDF/全文链接也可能含 DOI
        doi = _extract_doi_from_text(link, block)
        ranked.append(
            (
                rank,
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
                },
            )
        )

    ranked.sort(key=lambda x: x[0])
    for _, rec in ranked:
        records.append(rec)
        if len(records) >= max_results:
            break
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
    names: list[str] = []
    for a in item.get("author") or []:
        fam = (a.get("family") or "").strip()
        given = (a.get("given") or "").strip()
        initials = "".join(p[0].upper() for p in re.split(r"[\s\-]+", given) if p)
        if fam and initials:
            names.append(f"{fam} {initials}")
        elif fam:
            names.append(fam)
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


def _crossref_lookup_by_doi(
    doi: str,
    *,
    timeout_seconds: int = 20,
) -> Optional[dict[str, Any]]:
    d = (doi or "").strip().lower()
    if not d.startswith("10."):
        return None
    headers = {
        "Accept": "application/json",
        "User-Agent": "aicheckword-literature/1.0 (metadata enrichment)",
    }
    try:
        resp = _outbound_get(
            f"{CROSSREF_BASE}/{quote_plus(d)}",
            headers=headers,
            timeout_seconds=timeout_seconds,
            force_proxy=False,
            retries=2,
        )
        data = resp.json()
    except Exception:
        return None
    msg = (data or {}).get("message")
    return msg if isinstance(msg, dict) else None


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
    headers = {
        "Accept": "application/json",
        "User-Agent": "aicheckword-literature/1.0 (metadata enrichment)",
    }
    try:
        resp = _outbound_get(
            CROSSREF_BASE,
            params=params,
            headers=headers,
            timeout_seconds=timeout_seconds,
            force_proxy=False,
            retries=2,
        )
        data = resp.json()
    except Exception:
        return None
    items = ((data or {}).get("message") or {}).get("items") or []
    want_key = _norm_title_key(t)
    if not want_key:
        return None
    for it in items:
        titles = it.get("title") or []
        cand = titles[0] if titles else ""
        ck = _norm_title_key(cand)
        if not ck:
            continue
        # 强标题匹配，避免张冠李戴
        if ck == want_key:
            return it
        if len(want_key) >= 16 and (ck.startswith(want_key[:24]) or want_key.startswith(ck[:24])):
            return it
    return None


def _apply_crossref(rec: dict[str, Any], item: dict[str, Any]) -> None:
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

    cur_journal = (rec.get("journal") or "").strip()
    if journal and (not cur_journal or _is_truncated_field(cur_journal)):
        rec["journal"] = journal
    elif not (rec.get("journal") or "").strip() and publisher:
        rec["journal"] = publisher

    cur_vip = (rec.get("volume_issue_pages") or "").strip()
    if vip and (not cur_vip or _is_truncated_field(cur_vip)):
        rec["volume_issue_pages"] = vip

    if doi:
        rec["doi"] = doi
    if year and (not (rec.get("year") or "").strip() or _is_truncated_field(rec.get("year"))):
        rec["year"] = year
    if year:
        rec["pub_date"] = year

    cur_authors = (rec.get("authors") or "").strip()
    if authors and (not cur_authors or _is_truncated_field(cur_authors) or len(cur_authors) < 4):
        rec["authors"] = authors

    # 标题被 Scholar 截断时用 Crossref 全称覆盖
    cur_title = (rec.get("title") or "").strip()
    if full_title and (not cur_title or _is_truncated_field(cur_title)):
        rec["title"] = full_title


def _enrich_one_record(rec: dict[str, Any]) -> None:
    """单条补全（供线程池调用）。"""
    existing_doi = (rec.get("doi") or "").strip()
    # URL 里的 DOI 可能被 Scholar 截断（如 phy2.1482 实为 phy2.14827），
    # 仅作查验线索，验证不通过就不落库。
    url_doi = existing_doi or _extract_doi_from_text(
        rec.get("source_url") or "", rec.get("journal") or ""
    )

    need = (
        _is_truncated_field(rec.get("journal"))
        or _is_truncated_field(rec.get("authors"))
        or _is_truncated_field(rec.get("title"))
        or not (rec.get("volume_issue_pages") or "").strip()
        or not existing_doi
    )
    if not need:
        return

    item = None
    if url_doi:
        item = _crossref_lookup_by_doi(url_doi, timeout_seconds=8)
        if item is not None and not existing_doi:
            # DOI 经 Crossref 校验通过，可信
            rec["doi"] = url_doi
    # DOI 查不到（或本就截断/缺失）时按标题回查，Crossref 命中会写回权威 DOI
    if item is None:
        item = _crossref_lookup(rec.get("title") or "", timeout_seconds=8)
    if item:
        _apply_crossref(rec, item)


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
            _is_truncated_field(rec.get("journal"))
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
_SCHOLAR_NEXT_RE = re.compile(
    r'<button[^>]+aria-label="Next"[^>]*>|<a[^>]+aria-label="Next"[^>]*>|'
    r'<td[^>]*class="[^"]*gs_n[^"]*"[^>]*>.*?(?:Next|下一页)',
    re.I | re.S,
)


def _parse_scholar_total_found(html_text: str) -> int:
    m = _SCHOLAR_TOTAL_RE.search(html_text or "")
    if not m:
        return 0
    raw = m.group(1) or m.group(2) or ""
    try:
        return int(raw.replace(",", ""))
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
    for warm_url in (
        f"https://scholar.google.com/?hl={quote_plus(lang)}",
        f"https://scholar.google.com/scholar_settings?hl={quote_plus(lang)}",
    ):
        try:
            resp = _outbound_get(
                warm_url,
                headers=headers,
                timeout_seconds=timeout_seconds,
                force_proxy=True,
                cookies=cookie_map,
                retries=2,
            )
            try:
                cookie_map.update({k: v for k, v in resp.cookies.items()})
            except Exception:
                pass
            time.sleep(1.5 + random.uniform(0.5, 1.5))
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
) -> tuple[list[dict[str, Any]], int]:
    """返回 (records, total_found)。

    策略（保证数据全面、排序对齐浏览器）：
    - 默认 ``hl=zh-CN`` + ``as_sdt=0,5``，与浏览器结果页参数一致；
    - **逐页顺序**抓取（每页 10 条），按 ``start`` / ``data-rp`` 保留浏览器顺序；
    - 每页之间**暂停数秒**（3–6s）降低限流；
    - 空页允许**重试+退避**；
    - 字段截断时用 Crossref（优先 DOI）覆盖补全。
    """
    q = (query or "").strip()
    if not q:
        return [], 0
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
    seen: set[str] = set()
    total_found = 0
    first_url = ""
    empty_pages = 0
    hard_page_cap = 120
    page_no = 0
    start = offset

    while len(all_records) < want and page_no < hard_page_cap:
        if total_found and start >= total_found:
            break
        # 抓取总时限：超时则优雅停止，返回已抓部分（可点继续检索续抓）
        if time.monotonic() > fetch_deadline:
            break
        if page_no == 0:
            time.sleep(0.8 + random.uniform(0.4, 1.2))
        else:
            # 翻页间隔模拟真人（抬高下限是抗 429 关键），偶尔更长停顿
            delay = random.uniform(_SCHOLAR_PAGE_DELAY_MIN, _SCHOLAR_PAGE_DELAY_MAX)
            if page_no % 5 == 0:
                delay += random.uniform(4.0, 9.0)
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
                    ) from exc
                raise ScholarFetchError(
                    f"Scholar 请求失败（已重试）：{exc}",
                    partial_records=[],
                    total_found=total_found,
                ) from exc

            html_text = ""
            if not rate_limited and resp is not None:
                if not first_url:
                    first_url = url
                # 编码已在出站处按 UTF-8 设定
                html_text = resp.text or ""
                if not total_found:
                    parsed_total = _parse_scholar_total_found(html_text)
                    if parsed_total:
                        total_found = parsed_total
                body_l = html_text.lower()
                robot = (
                    "please show you're not a robot" in body_l
                    or "unusual traffic" in body_l
                )
                if _is_scholar_rate_limited(resp=resp) or robot:
                    rate_limited = True

            if rate_limited:
                # 429/人机验证：真人此时会等一会再试。对同一页做长退避重试，
                # 而不是立刻中断整次检索（这正是之前卡在 ~44 条的原因）。
                if rl_retry < _SCHOLAR_RL_RETRIES and time.monotonic() < fetch_deadline:
                    rl_retry += 1
                    wait = min(90.0, 18.0 * rl_retry + random.uniform(4.0, 12.0))
                    time.sleep(wait)
                    # 退避后重新热身一次 cookie，进一步降低持续限流
                    if rl_retry == _SCHOLAR_RL_RETRIES - 1:
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
                    if not captured_rl.search_url:
                        captured_rl.search_url = first_url
                    raise captured_rl
                _raise_rate_limited(
                    resp=resp,
                    search_url=first_url or url,
                    partial_records=all_records,
                    total_found=total_found,
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
            if empty_pages >= 2 and not (total_found and start + page_size < total_found):
                break
            start += page_size
            continue
        empty_pages = 0

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
            all_records.append(rec)
            if len(all_records) >= want:
                break
        start += page_size
        # 已取满「想要」或已达到库内命中总数 → 结束
        if total_found and len(all_records) >= min(want, total_found):
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
    return result, total_found


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
) -> dict[str, Any]:
    from config.cursor_overrides import get_llm_http_proxy
    from src.core.scholar_captcha import create_scholar_captcha_session

    srcs = [str(s or "").strip().lower() for s in (sources or []) if str(s or "").strip()]
    srcs = [s for s in srcs if s in ("pubmed", "scholar")]
    if not srcs:
        raise ValueError("sources 至少包含 pubmed 或 scholar 之一")

    jobs: dict[str, Callable[[], tuple[list[dict[str, Any]], int]]] = {}
    if "pubmed" in srcs:
        jobs["pubmed"] = lambda: search_pubmed(
            query=query,
            start_year=start_year,
            end_year=end_year,
            max_results=max_results_per_source,
        )
    if "scholar" in srcs:
        jobs["scholar"] = lambda: search_scholar(
            query=query,
            start_year=start_year,
            end_year=end_year,
            max_results=max_results_per_source,
            captcha_session_id=scholar_captcha_session_id,
            sort_by=scholar_sort_by,
            start_offset=scholar_start_offset,
            hl=(scholar_hl or "zh-CN").strip() or "zh-CN",
        )

    details: list[dict[str, Any]] = []
    aggregated: list[dict[str, Any]] = []
    captcha_session_id = ""
    needs_captcha = False
    with ThreadPoolExecutor(max_workers=min(2, len(jobs))) as pool:
        futs = {pool.submit(fn): source for source, fn in jobs.items()}
        for fut in as_completed(futs):
            source = futs[fut]
            t0 = time.perf_counter()
            total_found = 0
            try:
                records, total_found = fut.result()
                err = ""
            except ScholarRateLimitedError as exc:
                records = list(getattr(exc, "partial_records", None) or [])
                total_found = int(getattr(exc, "total_found", 0) or 0)
                err = str(exc)
                if records:
                    err = f"{err}（已保留本源前 {len(records)} 条；验证后可继续补全翻页）"
                needs_captcha = True
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
    }
