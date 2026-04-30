"""
二进制 Word 97-2003（.doc / .dot，非 OOXML）→ .docx，供仅支持 OOXML 的就地 patch 等链路使用。

策略（不设必选商业依赖，按环境能力回退）：
1) LibreOffice / soffice 无头转换（PATH 中可找到则尝试，跨平台）
2) Windows 上可选 Microsoft Word COM（pywin32，requirements 已声明 win32 条件依赖）
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Tuple


def convert_binary_word_to_docx(*, src_path: str | Path, dst_path: str | Path, timeout_sec: int = 180) -> Tuple[bool, str]:
    """
    将 .doc / .dot（假定二进制 OLE）转为 .docx，写入 dst_path（可覆盖）。

    返回 (成功, 失败说明)。
    """
    src = Path(src_path).resolve()
    dst = Path(dst_path).resolve()
    if not src.is_file():
        return False, f"源文件不存在：{src}"
    suf = src.suffix.lower()
    if suf not in (".doc", ".dot"):
        return False, f"不支持的后缀：{suf}（此处仅处理 .doc / .dot）"
    try:
        with src.open("rb", buffering=0) as bf:
            head = bf.read(4)
    except OSError as e:
        return False, f"读取源文件失败：{e}"
    if head == b"PK\x03\x04":
        return False, "源文件实为 OOXML（zip），不应走二进制转换"

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            dst.unlink()
        except OSError:
            pass

    if _convert_via_libreoffice(src, dst, timeout_sec=timeout_sec):
        return True, ""
    if _convert_via_msword_com(src, dst):
        return True, ""
    return False, (
        "未找到可用的 doc→docx 转换方式：已尝试 LibreOffice（soffice/libreoffice）与 Windows Word（pywin32）。"
        "请安装 LibreOffice 或 Microsoft Word，或在本机用 Word 另存为 .docx 后再上传。"
    )


def _convert_via_libreoffice(src: Path, dst: Path, *, timeout_sec: int) -> bool:
    out_dir = dst.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = out_dir / f"{src.stem}.docx"

    if sys.platform == "win32":
        candidates = ("soffice.com", "soffice", "libreoffice.com", "libreoffice")
    else:
        candidates = ("libreoffice", "soffice")

    for exe in candidates:
        try:
            proc = subprocess.run(
                [
                    exe,
                    "--headless",
                    "--nologo",
                    "--nodefault",
                    "--nolockcheck",
                    "--convert-to",
                    "docx",
                    "--outdir",
                    str(out_dir),
                    str(src),
                ],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except FileNotFoundError:
            continue
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode != 0:
            continue
        if not expected.is_file():
            continue
        try:
            if expected.resolve() == dst.resolve():
                return dst.is_file()
            shutil.move(str(expected), str(dst))
        except OSError:
            if dst.is_file():
                return True
            continue
        return dst.is_file()
    return False


def _convert_via_msword_com(src: Path, dst: Path) -> bool:
    if sys.platform != "win32":
        return False
    try:
        import win32com.client  # type: ignore[import-untyped]
    except ImportError:
        return False

    wd_format_xml_document = 12
    word = None
    src_abs = str(src.resolve()).replace("/", "\\")
    out_abs = str(dst.resolve()).replace("/", "\\")
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        try:
            word.DisplayAlerts = 0
        except Exception:
            pass
        doc = word.Documents.Open(src_abs, ReadOnly=True, ConfirmConversions=False)
        try:
            doc.SaveAs2(out_abs, FileFormat=wd_format_xml_document)
        finally:
            try:
                doc.Close(False)
            except Exception:
                pass
        try:
            word.Quit()
        except Exception:
            pass
        word = None
        return dst.is_file()
    except Exception:
        try:
            if word is not None:
                word.Quit()
        except Exception:
            pass
        try:
            if dst.exists():
                dst.unlink()
        except OSError:
            pass
        return False
