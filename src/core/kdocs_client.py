"""金山文档客户端：支持两种模式拉取文档内容用于审核
1. 直接下载+本地解析（无需配置开放平台，只需文档下载直链）
2. 开放平台 API 提取纯文本（需 KDOCS_APP_ID / KDOCS_APP_KEY）
"""

import base64
import hashlib
import json
import os
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Optional

from config import settings


def has_api_credentials() -> bool:
    return bool((settings.kdocs_app_id or "").strip() and (settings.kdocs_app_key or "").strip())


def _wps2_sign(method: str, uri: str, body: bytes, content_type: str, date: str) -> str:
    """WPS-2 签名：sha1(app_key + Content-Md5 + Content-Type + DATE)，GET 时 Body 用 URI 的 MD5。"""
    app_key = (settings.kdocs_app_key or "").encode("utf-8")
    if method.upper() == "GET" or not body:
        content_md5 = hashlib.md5(uri.encode("utf-8")).hexdigest().lower()
    else:
        content_md5 = hashlib.md5(body).hexdigest().lower()
    sign_str = app_key.decode("utf-8") + content_md5 + content_type + date
    sig = hashlib.sha1(sign_str.encode("utf-8")).hexdigest().lower()
    return f"WPS-2:{settings.kdocs_app_id}:{sig}"


def download_file_from_url(download_url: str, timeout: int = 60) -> bytes:
    """从直链下载文件二进制内容。"""
    req = urllib.request.Request(download_url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _guess_filename_from_url(url: str, fallback: str = "document") -> str:
    """从 URL 路径猜测文件名；猜不到则用 fallback。"""
    from urllib.parse import urlparse, unquote
    parsed = urlparse(url)
    path = unquote(parsed.path or "")
    basename = Path(path).name if path else ""
    if basename and "." in basename:
        return basename
    return fallback


def fetch_plaintext_local(download_url: str, filename: str = "") -> str:
    """
    直接下载文件 → 本地解析提取文本。无需金山文档开放平台 API。
    支持 docx / pdf / xlsx / xls / txt / md 等本系统已支持的格式。
    """
    fn = (filename or "").strip() or _guess_filename_from_url(download_url, "document.docx")
    suffix = Path(fn).suffix.lower()
    if not suffix:
        suffix = ".docx"
        fn = fn + suffix

    raw = download_file_from_url(download_url)
    if not raw:
        raise RuntimeError("下载文件为空")

    tmp_dir = tempfile.mkdtemp(prefix="kdocs_")
    tmp_path = os.path.join(tmp_dir, fn)
    try:
        with open(tmp_path, "wb") as f:
            f.write(raw)
        from .document_loader import load_single_file
        docs = load_single_file(tmp_path)
        return "\n".join(d.page_content for d in docs)
    finally:
        try:
            os.remove(tmp_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass


def fetch_plaintext_from_url(download_url: str, filename: str, password: Optional[str] = None) -> str:
    """
    拉取文档纯文本。自动选择方式：
    - 有开放平台 API 凭据 → 走平台接口（支持密码保护文档）
    - 无凭据 → 直接下载文件+本地解析（无需开发者配置）
    """
    if not has_api_credentials():
        return fetch_plaintext_local(download_url, filename)

    from email.utils import formatdate
    host = "developer.kdocs.cn"
    content_type = "application/json"
    date = formatdate(timeval=time.time(), localtime=False, usegmt=True)
    body = json.dumps({"url": download_url, "filename": filename, **({"password": password} if password else {})}).encode("utf-8")
    uri = "/api/v1/openapi/office/extract/plaintext"
    auth = _wps2_sign("POST", uri, body, content_type, date)
    req = urllib.request.Request(
        f"https://{host}{uri}",
        data=body,
        method="POST",
        headers={
            "Date": date,
            "Content-Md5": hashlib.md5(body).hexdigest().lower(),
            "Content-Type": content_type,
            "Authorization": auth,
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("code") != 0:
        raise RuntimeError(f"金山文档提取失败：{data.get('result', data)}")
    task_id = (data.get("data") or {}).get("task_id")
    if not task_id:
        raise RuntimeError("金山文档未返回 task_id")
    for _ in range(60):
        time.sleep(1)
        uri_get = f"/api/v1/openapi/office/tasks/{task_id}"
        date = formatdate(timeval=time.time(), localtime=False, usegmt=True)
        auth = _wps2_sign("GET", uri_get, b"", content_type, date)
        req_get = urllib.request.Request(
            f"https://{host}{uri_get}",
            headers={
                "Date": date,
                "Content-Md5": hashlib.md5(uri_get.encode("utf-8")).hexdigest().lower(),
                "Content-Type": content_type,
                "Authorization": auth,
            },
        )
        with urllib.request.urlopen(req_get, timeout=15) as r:
            out = json.loads(r.read().decode("utf-8"))
        if out.get("code") != 0:
            raise RuntimeError(f"金山文档任务查询失败：{out}")
        d = out.get("data") or {}
        status = d.get("status")
        if status == "success":
            result = d.get("result") or {}
            b64 = result.get("base_64_text")
            if not b64:
                return ""
            return base64.b64decode(b64).decode("utf-8", errors="replace")
        if status == "failed":
            raise RuntimeError(f"金山文档提取失败：{d.get('message', 'unknown')}")
    raise RuntimeError("金山文档提取超时")
