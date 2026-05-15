# -*- coding: utf-8 -*-
"""项目下拉展示文案：名称、ID、项目编号、产品名称、注册国家、注册类别。"""
from __future__ import annotations

from typing import Any, Mapping


def format_project_option_label(p: Mapping[str, Any] | None) -> str:
    """与 Streamlit「选择项目」及集成侧 project label 一致。"""
    try:
        nm = str((p or {}).get("name") or "").strip() or "未命名"
    except Exception:
        nm = "未命名"
    try:
        pid = int((p or {}).get("id") or 0)
    except Exception:
        pid = 0
    pc = ""
    try:
        pc = str((p or {}).get("project_code") or "").strip()
    except Exception:
        pc = ""
    suf = f" · {pc}" if pc else ""
    head = f"{nm} (ID:{pid}){suf}"

    prod = ""
    try:
        prod = str((p or {}).get("product_name") or "").strip()
    except Exception:
        prod = ""
    if not prod:
        try:
            prod = str((p or {}).get("product_name_en") or "").strip()
        except Exception:
            prod = ""

    rcn = ""
    rce = ""
    try:
        rcn = str((p or {}).get("registration_country") or "").strip()
    except Exception:
        rcn = ""
    try:
        rce = str((p or {}).get("registration_country_en") or "").strip()
    except Exception:
        rce = ""
    if rcn and rce and rcn != rce:
        cshow = f"{rcn} / {rce}"
    elif rcn:
        cshow = rcn
    else:
        cshow = rce

    reg_type = ""
    try:
        reg_type = str((p or {}).get("registration_type") or "").strip()
    except Exception:
        reg_type = ""

    extras: list[str] = []
    if prod:
        extras.append(f"产品:{prod}")
    if cshow:
        extras.append(f"国家:{cshow}")
    if reg_type:
        extras.append(f"类别:{reg_type}")
    if not extras:
        return head
    return f"{head} | " + " | ".join(extras)
