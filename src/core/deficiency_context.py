# -*- coding: utf-8 -*-
"""发补修正意见上下文：按国家+类别+发补日期门槛注入下游。"""
from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional

from src.core.deficiency_store import list_injectable_deficiency_records


def _as_date(val) -> Optional[date]:
    if val is None:
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    try:
        return date.fromisoformat(str(val).strip()[:10])
    except ValueError:
        return None


def _type_label(t: str) -> str:
    return {
        "registration_review": "注册审评发补",
        "type_testing": "体考发补",
    }.get(str(t or ""), str(t or ""))


def _priority_label(p: str) -> str:
    return {"high": "高", "medium": "中", "low": "低"}.get(str(p or ""), str(p or ""))


def _status_label(s: str) -> str:
    return {"open": "未完成", "done": "已完成"}.get(str(s or ""), str(s or ""))


def build_deficiency_lessons_context(
    collection: str,
    *,
    as_of_date: date | str | datetime | None = None,
    registration_country: str | None = None,
    registration_category: str | None = None,
    deficiency_types: List[str] | None = None,
    query_text: str = "",
    top_k: int = 8,
) -> str:
    """构建【发补修正意见】区块。

    硬条件：同一 collection + 注册国家 + 注册类别 + issued_on <= as_of_date。
    open 与 done 均注入；不按所属项目过滤。
    """
    country = (registration_country or "").strip()
    category = (registration_category or "").strip()
    if not country or not category:
        return ""
    as_of = _as_date(as_of_date) or date.today()
    rows = list_injectable_deficiency_records(
        collection,
        registration_country=country,
        registration_category=category,
        as_of_date=as_of,
    )
    if deficiency_types:
        allow = {str(x).strip() for x in deficiency_types if str(x).strip()}
        if allow:
            rows = [r for r in rows if str(r.get("deficiency_type") or "") in allow]
    if not rows:
        return ""

    q = (query_text or "").strip().lower()
    if q:
        scored = []
        for r in rows:
            blob = " ".join(
                [
                    str(r.get("opinion_text") or ""),
                    str(r.get("remediation_plan") or ""),
                    str(r.get("deficiency_source") or ""),
                ]
            ).lower()
            score = sum(1 for tok in q.replace("，", " ").replace(",", " ").split() if tok and tok in blob)
            scored.append((score, r))
        scored.sort(key=lambda x: (-x[0],))
        rows = [r for _, r in scored]

    rows = rows[: max(1, min(int(top_k or 8), 20))]
    lines = [
        "【发补修正意见】（发补日期≤操作日；同一注册国家+同一注册类别；含未完成与已完成）",
        f"匹配：国家={country} | 类别={category} | 操作日={as_of.isoformat()} | 命中={len(rows)}",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"{i}. [{_type_label(r.get('deficiency_type'))}] "
            f"发补日={str(r.get('issued_on') or '')[:10]} | "
            f"状态={_status_label(r.get('remediation_status'))} | "
            f"优先级={_priority_label(r.get('priority'))} | "
            f"来源={str(r.get('deficiency_source') or '—')}"
        )
        opinion = str(r.get("opinion_text") or "").strip()
        plan = str(r.get("remediation_plan") or "").strip()
        if opinion:
            lines.append(f"   意见：{opinion[:1200]}")
        if plan:
            lines.append(f"   方案/修正要点：{plan[:1200]}")
        lines.append(
            "   要求：后续初稿/审核/修改/翻译/任务清单须避免重复上述问题；"
            "不得把整改前写法当模板；不得编造法规条款；追溯编号须与输入逐字一致。"
        )
    lines.append(
        "使用约束：以上仅作历史发补经验参考，须与当前项目资料与审核点清单交叉核对。"
    )
    return "\n".join(lines)
