"""上传审核场景下的「展示用文件名」与临时文件名校验（NamedTemporaryFile 等）。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Python tempfile.NamedTemporaryFile 在 Windows 上常见：tmp + 随机字符 + 后缀
_TEMP_UPLOAD_BASE_RE = re.compile(r"^tmp[a-z0-9]{4,24}\.[^.]+$", re.IGNORECASE)


def is_probable_temp_upload_basename(name: str) -> bool:
    """是否为系统生成的临时文件名（仅根据 basename 启发式判断）。"""
    if not name:
        return False
    base = Path(str(name).replace("\\", "/")).name
    return bool(_TEMP_UPLOAD_BASE_RE.match(base))


def effective_audit_report_display_name(
    report: Optional[Dict[str, Any]],
    *,
    db_file_name: str = "",
) -> str:
    """
    从报告 JSON + 库表 file_name 推断应对用户展示的文档名。
    优先非临时名的 original_filename / file_name；否则回退库表列。
    """
    rep = report if isinstance(report, dict) else {}
    fn_col = (db_file_name or "").strip()
    o = (rep.get("original_filename") or "").strip()
    f = (rep.get("file_name") or "").strip()
    for cand in (o, f, fn_col):
        if not cand:
            continue
        bn = Path(cand.replace("\\", "/")).name
        if not is_probable_temp_upload_basename(bn):
            return cand
    return o or f or fn_col or ""


def sanitize_audit_report_dict(
    report: Optional[Dict[str, Any]],
    *,
    db_file_name: str = "",
) -> None:
    """
    就地修正报告内展示用文件名：顶层 file_name/original_filename、related_doc_names 与各审核点 modify_docs 中的临时名。
    仅用于内存展示/导出前，不写回数据库（除非调用方显式 save）。
    """
    if not isinstance(report, dict):
        return
    disp = effective_audit_report_display_name(report, db_file_name=db_file_name)
    if not disp:
        return
    fn_now = (report.get("file_name") or "").strip()
    if is_probable_temp_upload_basename(Path(fn_now.replace("\\", "/")).name) or not fn_now:
        report["file_name"] = disp
    o_now = (report.get("original_filename") or "").strip()
    if is_probable_temp_upload_basename(Path(o_now.replace("\\", "/")).name) or not o_now:
        report["original_filename"] = disp

    rdn = report.get("related_doc_names")
    if isinstance(rdn, list):
        new_r: List[str] = []
        seen_r = set()
        for x in rdn:
            if x is None:
                continue
            sx = str(x).strip()
            if not sx:
                continue
            bn = Path(sx.replace("\\", "/")).name
            if is_probable_temp_upload_basename(bn):
                sx = disp
            if sx not in seen_r:
                seen_r.add(sx)
                new_r.append(sx)
        if new_r:
            report["related_doc_names"] = new_r

    points = report.get("audit_points")
    if not isinstance(points, list):
        return
    for p in points:
        if not isinstance(p, dict):
            continue
        md_raw = p.get("modify_docs")
        if md_raw is None:
            continue
        if isinstance(md_raw, str):
            md_list = [md_raw]
        else:
            md_list = list(md_raw) if isinstance(md_raw, (list, tuple)) else []
        new_md: List[str] = []
        seen = set()
        for m in md_list:
            s = str(m).strip()
            if not s:
                continue
            bn = Path(s.replace("\\", "/")).name
            if is_probable_temp_upload_basename(bn):
                s = disp
            if s not in seen:
                seen.add(s)
                new_md.append(s)
        p["modify_docs"] = new_md
