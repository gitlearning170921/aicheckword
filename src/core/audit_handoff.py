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


def build_audit_point_coverage_report(
    immediate_points: List[Dict[str, Any]],
    patch_report: Dict[str, Any],
) -> Dict[str, Any]:
    """
    对照「立即修改」审核点与 patch.report 中的 changes/skipped，生成逐点覆盖报告。
    status: modified | not_addressed | partial（有 change 但关联 operation 也被 skip）
    """
    changes = list((patch_report or {}).get("changes") or [])
    skipped = list((patch_report or {}).get("skipped") or [])

    ref_to_changes: Dict[str, List[Dict[str, Any]]] = {}
    ref_to_skips: Dict[str, List[Dict[str, Any]]] = {}

    for ch in changes:
        if not isinstance(ch, dict):
            continue
        refs = ch.get("audit_point_refs") or []
        if isinstance(refs, list) and refs:
            for r in refs:
                rk = str(r).strip()
                if rk:
                    ref_to_changes.setdefault(rk, []).append(ch)
        else:
            ref_to_changes.setdefault("__unattributed__", []).append(ch)

    for sk in skipped:
        if not isinstance(sk, dict):
            continue
        op = sk.get("op") if isinstance(sk.get("op"), dict) else {}
        refs = op.get("audit_point_refs") or []
        if isinstance(refs, list) and refs:
            for r in refs:
                rk = str(r).strip()
                if rk:
                    ref_to_skips.setdefault(rk, []).append(sk)
        else:
            ref_to_skips.setdefault("__unattributed__", []).append(sk)

    points_out: List[Dict[str, Any]] = []
    modified_n = 0
    not_addr_n = 0
    partial_n = 0

    for pt in immediate_points or []:
        if not isinstance(pt, dict):
            continue
        ref = str(pt.get("audit_point_ref") or "").strip()
        chs = ref_to_changes.get(ref) or []
        sks = ref_to_skips.get(ref) or []

        if chs and sks:
            status = "partial"
            partial_n += 1
        elif chs:
            status = "modified"
            modified_n += 1
        else:
            status = "not_addressed"
            not_addr_n += 1

        change_details: List[Dict[str, Any]] = []
        for ch in chs:
            change_details.append(
                {
                    "type": ch.get("type"),
                    "anchor": str(ch.get("anchor") or "")[:200],
                    "before": str(ch.get("before") or "")[:2000],
                    "after": str(ch.get("after") or "")[:2000],
                    "note": str(ch.get("note") or "")[:500],
                }
            )

        skip_reasons: List[str] = []
        for sk in sks:
            reason = str(sk.get("reason") or "").strip()
            if reason:
                skip_reasons.append(reason)

        not_addressed_reason = ""
        if not chs:
            if sks:
                not_addressed_reason = "；".join(dict.fromkeys(skip_reasons[:8])) or "关联 operation 被跳过"
            else:
                not_addressed_reason = (
                    "模型 PATCH 未生成带本 ref 的可执行 operation，或 operation 未命中文档锚点"
                )

        points_out.append(
            {
                "audit_point_ref": ref,
                "status": status,
                "category": pt.get("category"),
                "location": pt.get("location"),
                "description": pt.get("description"),
                "suggestion": pt.get("suggestion"),
                "changes": change_details,
                "skip_reasons": skip_reasons,
                "not_addressed_reason": not_addressed_reason,
            }
        )

    unattributed_changes = len(ref_to_changes.get("__unattributed__") or [])
    unattributed_skips = len(ref_to_skips.get("__unattributed__") or [])

    return {
        "summary": {
            "total_immediate_points": len(points_out),
            "modified": modified_n,
            "not_addressed": not_addr_n,
            "partial": partial_n,
            "unattributed_changes": unattributed_changes,
            "unattributed_skips": unattributed_skips,
        },
        "points": points_out,
    }


def format_audit_point_coverage_markdown(coverage: Dict[str, Any], *, file_name: str = "") -> str:
    """将 audit_point_coverage 格式化为可读 Markdown（写入 *.audit_modify.log.md）。"""
    if not isinstance(coverage, dict):
        return ""
    sm = coverage.get("summary") or {}
    lines: List[str] = [
        "# 审核后修改 — 审核点覆盖日志",
    ]
    if file_name:
        lines.append(f"\n**目标文件**：{file_name}")
    lines.append(
        f"\n**汇总**：共 {sm.get('total_immediate_points', 0)} 个立即修改点 · "
        f"已修改 {sm.get('modified', 0)} · "
        f"部分 {sm.get('partial', 0)} · "
        f"未落实 {sm.get('not_addressed', 0)}"
    )
    if sm.get("unattributed_changes") or sm.get("unattributed_skips"):
        lines.append(
            f"\n（另有 {sm.get('unattributed_changes', 0)} 条变更、"
            f"{sm.get('unattributed_skips', 0)} 条跳过未绑定 audit_point_ref）"
        )

    for i, pt in enumerate(coverage.get("points") or [], start=1):
        if not isinstance(pt, dict):
            continue
        ref = pt.get("audit_point_ref") or "?"
        st = pt.get("status") or "not_addressed"
        st_zh = {"modified": "已修改", "partial": "部分修改", "not_addressed": "未修改"}.get(
            str(st), str(st)
        )
        lines.append(f"\n## {i}. [{ref}] {st_zh}")
        if pt.get("category"):
            lines.append(f"- **类别**：{pt.get('category')}")
        if pt.get("location"):
            lines.append(f"- **位置**：{pt.get('location')}")
        if pt.get("description"):
            lines.append(f"- **问题**：{pt.get('description')}")
        if pt.get("suggestion"):
            lines.append(f"- **建议**：{pt.get('suggestion')}")

        chs = pt.get("changes") or []
        if chs:
            lines.append("\n### 已写入的修改")
            for j, ch in enumerate(chs, start=1):
                if not isinstance(ch, dict):
                    continue
                lines.append(f"\n#### 变更 {j}（{ch.get('type') or 'op'}）")
                if ch.get("anchor"):
                    lines.append(f"- 锚点：{ch.get('anchor')}")
                before = str(ch.get("before") or "").strip()
                after = str(ch.get("after") or "").strip()
                if before:
                    lines.append(f"- **修改前**：\n```\n{before[:1500]}\n```")
                if after:
                    lines.append(f"- **修改后**：\n```\n{after[:1500]}\n```")
                if ch.get("note"):
                    lines.append(f"- 备注：{ch.get('note')}")

        if st != "modified":
            reason = pt.get("not_addressed_reason") or ""
            skips = pt.get("skip_reasons") or []
            lines.append("\n### 未完全落实原因")
            if reason:
                lines.append(f"- {reason}")
            for sr in skips[:6]:
                lines.append(f"- 跳过：{sr}")

    return "\n".join(lines).strip() + "\n"
