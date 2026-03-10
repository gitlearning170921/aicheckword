"""从安装包或 URL 识别系统功能，供按项目审核时与文档一致性核对。"""

import re
import zipfile
import tarfile
from pathlib import Path
from typing import Optional

from config import settings


# 用于 LLM 识别的提示词模板
SYSTEM_FUNCTIONALITY_PROMPT = """请根据以下内容，提炼并输出该软件/系统的**功能描述**，用于与注册文档（如说明书、技术要求）做一致性核对。

## 原始内容来源
{source_hint}

## 原始内容（文件列表、页面文本等）
{raw_content}

## 要求
1. 用简洁的中文列出主要功能模块、关键特性、界面/操作要点（与说明书、技术要求可对照的内容）。
2. 若为安装包：可根据文件路径、名称、可读文本推断软件类型与功能。
3. 若为网页/系统：可根据页面结构、文案、菜单推断系统功能。
4. 输出一段结构化文本，便于后续审核时核对待审文档中的功能描述是否与实际一致。
5. 不要输出与功能无关的说明，控制在 800 字以内。"""


def extract_package_info(file_path: str) -> str:
    """从安装包或压缩包中提取可用于识别系统功能的信息：文件列表 + 部分可读文本。"""
    path = Path(file_path)
    if not path.is_file():
        return f"文件不存在或不可读：{file_path}"

    suffix = path.suffix.lower()
    name = path.name
    lines = [f"安装包/文件：{name}", ""]

    # ZIP / APK (APK 本质是 ZIP)，使用正确编码避免文件名乱码
    if suffix == ".zip" or suffix == ".apk" or (suffix == ".gz" and name.lower().endswith(".apk.gz")):
        try:
            from .document_loader import _open_zip_with_encoding
            zf = _open_zip_with_encoding(path)
            try:
                namelist = zf.namelist()[:300]  # 限制数量
                lines.append("【文件列表】")
                lines.extend(namelist)
                # 尝试读取常见说明类文件
                for candidate in ("readme.txt", "README.TXT", "说明.txt", "version.txt", "AndroidManifest.xml"):
                    for n in namelist:
                        if n.lower().endswith(candidate.lower()) or candidate.lower() in n.lower():
                            try:
                                with zf.open(n) as f:
                                    raw = f.read(4096)
                                    text = raw.decode("utf-8", errors="ignore") or raw.decode("gbk", errors="ignore")
                                    if text.strip():
                                        lines.append("")
                                        lines.append(f"【{n} 摘要】")
                                        lines.append(text.strip()[:2000])
                            except Exception:
                                pass
                            break
            finally:
                zf.close()
        except Exception as e:
            lines.append(f"解压/读取失败：{e}")
        return "\n".join(lines)

    # TAR / TAR.GZ / TGZ
    if suffix in (".tar", ".gz", ".tgz") or ".tar." in name.lower():
        try:
            with tarfile.open(path, "r:*") as tf:
                names = tf.getnames()[:300]
                lines.append("【文件列表】")
                lines.extend(names)
                for member in tf.getmembers()[:50]:
                    if member.isfile() and member.size < 50000:
                        name_m = getattr(member, "name", None) or getattr(member, "path", "")
                        if any(x in name_m.lower() for x in ("readme", "说明", "version", ".txt", ".xml")):
                            try:
                                f = tf.extractfile(member)
                                if f:
                                    raw = f.read(4096)
                                    text = raw.decode("utf-8", errors="ignore") or raw.decode("gbk", errors="ignore")
                                    if text.strip():
                                        lines.append("")
                                        lines.append(f"【{name_m} 摘要】")
                                        lines.append(text.strip()[:2000])
                            except Exception:
                                pass
        except Exception as e:
            lines.append(f"解压/读取失败：{e}")
        return "\n".join(lines)

    # EXE / MSI 等：仅能提供文件名与大小
    if suffix in (".exe", ".msi", ".dmg", ".deb", ".rpm"):
        size_mb = path.stat().st_size / (1024 * 1024)
        lines.append(f"文件类型：{suffix}，大小约 {size_mb:.2f} MB。")
        lines.append("（无法直接解析二进制安装包内容，请在上传前补充说明文档，或使用「输入 URL」方式录入系统功能。）")
        return "\n".join(lines)

    return "\n".join(lines + ["未知格式，仅做占位。"])


def fetch_url_content(
    url: str,
    username: str = "",
    password: str = "",
    max_chars: int = 15000,
    captcha: str = "",
) -> str:
    """请求 URL 获取正文文本（可选 Basic 认证、可选验证码参数），用于识别系统功能。"""
    import urllib.parse
    import httpx

    if not url or not url.strip():
        return "未填写 URL。"

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    if captcha and captcha.strip():
        sep = "&" if "?" in url else "?"
        url = url + sep + "captcha=" + urllib.parse.quote(captcha.strip())

    auth = None
    if username and username.strip():
        auth = (username.strip(), (password or "").strip())

    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            resp = client.get(url, auth=auth)
            resp.raise_for_status()
            text = resp.text
    except Exception as e:
        return f"请求失败：{e}\n请检查 URL 是否可访问、是否需要登录（本功能仅支持 Basic 认证）。"

    # 简单去标签，保留可见文本
    text = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", text, flags=re.I)
    text = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()[:max_chars]

    # 检测是否为验证码页面（提示用户输入验证码后重试）
    captcha_keywords = ["验证码", "captcha", "请输入验证码", "图形验证码", "人机验证"]
    if len(text) < 800 and any(k in text for k in captcha_keywords):
        return "CAPTCHA_REQUIRED: 页面需要验证码，请在「验证码」输入框中填写后重试。\n" + (text[:500] or "")

    if not text:
        return "页面未解析出有效正文，可能需登录或为单页应用。请尝试在「自定义审核要求」中手动补充系统功能说明。"
    return text


def identify_system_functionality_with_llm(
    raw_content: str,
    source_hint: str,
    provider: Optional[str] = None,
) -> str:
    """用 LLM 根据原始内容生成系统功能描述。"""
    from config import settings as s

    prompt_text = SYSTEM_FUNCTIONALITY_PROMPT.format(
        source_hint=source_hint,
        raw_content=raw_content[:12000],
    )
    use_cursor = (provider or getattr(s, "provider", "") or "").strip().lower() == "cursor"
    if use_cursor:
        from .cursor_agent import complete_task
        return (complete_task(prompt_text) or "").strip()
    # Ollama / OpenAI
    from .reviewer import DocumentReviewer
    rev = DocumentReviewer()
    msg = rev.llm.invoke(prompt_text)
    return (getattr(msg, "content", str(msg)) or "").strip()
