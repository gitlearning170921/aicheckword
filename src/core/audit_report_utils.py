"""审核报告 JSON 的通用操作（供 API 与 Streamlit 共用，避免循环依赖）。"""

from typing import Any, Dict, List, Optional


def recount_severity(report: Dict[str, Any]) -> None:
    """根据 audit_points 重新统计各严重程度计数（弃用 deprecated 的点不计入）。"""
    counts = {"high": 0, "medium": 0, "low": 0, "info": 0}
    points = report.get("audit_points") or []
    for p in points:
        if p.get("deprecated"):
            continue
        s = (p.get("severity") or "info").lower()
        if s not in counts:
            s = "info"
        counts[s] = counts.get(s, 0) + 1
    report["high_count"] = counts["high"]
    report["medium_count"] = counts["medium"]
    report["low_count"] = counts["low"]
    report["info_count"] = counts["info"]
    report["total_points"] = len([p for p in points if not p.get("deprecated")])


def aggregate_batch_report_totals(parent: Dict[str, Any]) -> None:
    """根据子报告汇总批量报告的 total_points 与各严重程度计数。"""
    reports = parent.get("reports") or []
    parent["total_points"] = sum(r.get("total_points", 0) for r in reports)
    parent["high_count"] = sum(r.get("high_count", 0) for r in reports)
    parent["medium_count"] = sum(r.get("medium_count", 0) for r in reports)
    parent["low_count"] = sum(r.get("low_count", 0) for r in reports)
    parent["info_count"] = sum(r.get("info_count", 0) for r in reports)


def get_target_report_for_points(root: Dict[str, Any], sub_report_index: int = 0) -> Dict[str, Any]:
    """批量报告时返回第 sub_report_index 份子报告，否则返回 root。"""
    if root.get("batch") and root.get("reports"):
        reps: List[Dict[str, Any]] = root["reports"]
        if not reps:
            return root
        idx = max(0, min(sub_report_index, len(reps) - 1))
        return reps[idx]
    return root


def apply_point_field_updates(
    point: Dict[str, Any],
    description: Optional[str] = None,
    suggestion: Optional[str] = None,
    action: Optional[str] = None,
    modify_docs: Optional[List[str]] = None,
    severity: Optional[str] = None,
    regulation_ref: Optional[str] = None,
    location: Optional[str] = None,
    category: Optional[str] = None,
) -> None:
    if description is not None:
        point["description"] = description
    if suggestion is not None:
        point["suggestion"] = suggestion
    if action is not None:
        point["action"] = action
    if modify_docs is not None:
        point["modify_docs"] = modify_docs
    if severity is not None:
        point["severity"] = severity
    if regulation_ref is not None:
        point["regulation_ref"] = regulation_ref
    if location is not None:
        point["location"] = location
    if category is not None:
        point["category"] = category
