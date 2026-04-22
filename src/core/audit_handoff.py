"""审核「立即修改」点 → 文档生成侧交接（与 report_export 解耦，避免导入链/文件未同步问题）。"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

_SEVERITY_MAP = {"high": "高", "medium": "中", "low": "低", "info": "提示"}


def _sev_label(s: str) -> str:
    return _SEVERITY_MAP.get((s or "info").lower(), s or "")


def _default_action_for_severity(severity: str) -> str:
    severity = (severity or "info").lower()
    defaults = {"high": "立即修改", "medium": "立即修改", "low": "延期修改", "info": "无需修改"}
    return defaults.get(severity, "无需修改")


def _point_rows_from_single_subreport(
    sub: Dict[str, Any],
    *,
    get_default_action: Callable[[str], str],
    selected_refs: Optional[set],
    ref_batch_parent_id: Optional[int],
    ref_sub_index: Optional[int],
) -> List[Dict[str, Any]]:
    """从单份子报告抽取「立即修改」行；批次汇总时用 R{id}_S{si}_P{idx} 作为 audit_point_ref。"""
    if not isinstance(sub, dict):
        return []
    points = sub.get("audit_points") or []
    file_name = (
        str(sub.get("original_filename") or "").strip()
        or str(sub.get("file_name") or "").strip()
        or "未知文件"
    )
    try:
        rid = int(sub.get("id") or 0)
    except Exception:
        rid = 0
    selected = {str(x).strip() for x in (selected_refs or set()) if str(x).strip()}
    rows: List[Dict[str, Any]] = []
    for idx, p in enumerate(points):
        if not isinstance(p, dict):
            continue
        if p.get("deprecated"):
            continue
        sev = str(p.get("severity") or "info").lower()
        action = str(p.get("action") or get_default_action(sev)).strip()
        if action != "立即修改":
            continue
        if ref_batch_parent_id is not None and ref_sub_index is not None:
            ref = f"R{int(ref_batch_parent_id)}_S{int(ref_sub_index)}_P{idx}"
        else:
            ref = f"R{rid or 0}_P{idx}"
        if selected and ref not in selected:
            continue
        modify_docs = p.get("modify_docs") or []
        if isinstance(modify_docs, list) and any(str(x).strip() for x in modify_docs):
            targets = [str(x).strip() for x in modify_docs if str(x).strip()]
        else:
            targets = [file_name]
        rows.append(
            {
                "audit_point_ref": ref,
                "point_index": idx,
                "action": action,
                "severity": sev,
                "category": str(p.get("category") or "").strip(),
                "location": str(p.get("location") or "").strip(),
                "description": str(p.get("description") or "").strip(),
                "suggestion": str(p.get("suggestion") or "").strip(),
                "regulation_ref": str(p.get("regulation_ref") or "").strip(),
                "targets": targets,
                "source_file_name": file_name,
            }
        )
    return rows


def build_immediate_audit_point_records(
    report: Dict[str, Any],
    *,
    get_default_action: Optional[Callable[[str], str]] = None,
    selected_refs: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """
    从审核报告构建“立即修改”审核点结构化记录。
    - 单份报告：顶层 audit_points，ref 为 R{report_id}_P{idx}
    - 批次汇总：batch 且含 reports[] 时合并各子报告（含多文档一致性子报告），ref 为 R{汇总id}_S{子序号}_P{idx}
    """
    if not isinstance(report, dict):
        return []
    default_action_fn = get_default_action or _default_action_for_severity

    subs: List[Dict[str, Any]] = []
    if report.get("batch") and isinstance(report.get("reports"), list):
        for x in report.get("reports") or []:
            if isinstance(x, dict):
                subs.append(x)
    if subs:
        try:
            parent_id = int(report.get("id") or 0)
        except Exception:
            parent_id = 0
        merged: List[Dict[str, Any]] = []
        for si, sub in enumerate(subs):
            merged.extend(
                _point_rows_from_single_subreport(
                    sub,
                    get_default_action=default_action_fn,
                    selected_refs=selected_refs,
                    ref_batch_parent_id=parent_id,
                    ref_sub_index=si,
                )
            )
        return merged

    return _point_rows_from_single_subreport(
        report,
        get_default_action=default_action_fn,
        selected_refs=selected_refs,
        ref_batch_parent_id=None,
        ref_sub_index=None,
    )


def build_immediate_audit_remediation_by_target(
    report: Dict[str, Any],
    *,
    get_default_action: Optional[Callable[[str], str]] = None,
    selected_refs: Optional[set] = None,
    max_chars_per_target: int = 8000,
) -> Dict[str, Any]:
    """
    产出按目标文件聚合的“立即修改”文本与结构化列表：
    - points_by_target: {target_file: [point_record, ...]}
    - text_by_target: {target_file: "可直接注入提示词的文本"}
    - all_points: 所有命中的 point_record 列表
    """
    all_points = build_immediate_audit_point_records(
        report,
        get_default_action=get_default_action,
        selected_refs=selected_refs,
    )
    points_by_target: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_points:
        for target in row.get("targets") or []:
            t = str(target or "").strip()
            if not t:
                continue
            points_by_target.setdefault(t, []).append(row)

    text_by_target: Dict[str, str] = {}
    for target, rows in points_by_target.items():
        lines = [f"目标文件：{target}", "仅执行以下已选中审核点，禁止发散到未选中点："]
        for i, r in enumerate(rows, start=1):
            lines.append(
                f"{i}. [{r.get('audit_point_ref')}] "
                f"级别={_sev_label(r.get('severity') or 'info')} "
                f"类别={r.get('category') or '未分类'}"
            )
            if r.get("location"):
                lines.append(f"   位置：{r.get('location')}")
            if r.get("description"):
                lines.append(f"   问题：{r.get('description')}")
            if r.get("suggestion"):
                lines.append(f"   建议：{r.get('suggestion')}")
            if r.get("regulation_ref"):
                lines.append(f"   依据：{r.get('regulation_ref')}")
        text = "\n".join(lines).strip()
        cap = max(500, int(max_chars_per_target or 8000))
        if len(text) > cap:
            text = text[:cap] + "\n（其余审核点已截断，请优先处理高风险/中风险项）"
        text_by_target[target] = text
    return {
        "points_by_target": points_by_target,
        "text_by_target": text_by_target,
        "all_points": all_points,
    }
