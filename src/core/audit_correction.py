"""审核点纠正：误报 / 弃用 / 修订及反馈向量库入库（Streamlit 与 REST API 共用）。"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Literal, Optional, Tuple

from config import settings
from src.core.audit_report_utils import (
    aggregate_batch_report_totals,
    apply_point_field_updates,
    get_target_report_for_points,
    recount_severity,
)
from src.core.db import (
    OP_TYPE_CORRECTION,
    add_operation_log,
    get_current_model_info,
    save_audit_correction,
    update_audit_report,
)

CorrectionKind = Literal["revision", "false_positive", "deprecated"]
_logger = logging.getLogger(__name__)


def feed_correction_to_kb_impl(collection: str, file_name: str, corrected_point: dict) -> None:
    """同步写入反馈向量库（可在后台线程中调用）。"""
    from src.core.agent import ReviewAgent
    from src.core.langchain_compat import Document

    is_fp = corrected_point.get("correction_kind") == "false_positive" or bool(
        (corrected_point.get("false_positive_reason") or "").strip()
    )
    is_dep = bool(corrected_point.get("deprecated"))
    repl = corrected_point.get("replacement_point_added")

    if is_fp:
        feedback_kind = "false_positive"
        content = (
            f"[用户反馈·误报] 入库分类：{feedback_kind}（非审核点清单原文）\n"
            f"关联文件：{file_name}\n"
            f"人工标记为误报。原因：{corrected_point.get('false_positive_reason', '')}\n"
            f"原审核类别：{corrected_point.get('category', '')}\n"
            f"原问题描述：{corrected_point.get('description', '')}\n"
            f"原位置：{corrected_point.get('location', '')}\n"
            f"原法规依据：{corrected_point.get('regulation_ref', '')}\n"
            f"说明：若待审文档与上述原审核点在语义上等价，**不得**再输出该审核点。"
        )
    elif is_dep:
        feedback_kind = "deprecated_with_replacement" if repl else "deprecated"
        content = (
            f"[用户反馈·弃用审核点] 入库分类：{feedback_kind}（非审核点清单原文）\n"
            f"关联文件：{file_name}\n"
            f"弃用说明：{corrected_point.get('deprecation_note', '')}\n"
            f"原类别/描述摘要：{corrected_point.get('category', '')} / {corrected_point.get('description', '')}\n"
        )
        if isinstance(repl, dict):
            content += (
                f"\n同时新增替代审核点：类别={repl.get('category', '')}；描述={repl.get('description', '')}；"
                f"建议={repl.get('suggestion', '')}"
            )
    else:
        feedback_kind = "revision"
        content = (
            f"[用户反馈·修订后结论] 入库分类：{feedback_kind}（非审核点清单原文）\n"
            f"关联文件：{file_name}\n"
            f"类别：{corrected_point.get('category', '')}\n"
            f"严重程度：{corrected_point.get('severity', '')}\n"
            f"问题描述（修订后）：{corrected_point.get('description', '')}\n"
            f"法规依据：{corrected_point.get('regulation_ref', '')}\n"
            f"修改建议：{corrected_point.get('suggestion', '')}"
        )
    loc = corrected_point.get("location") or ""
    if loc:
        content += f"\n位置/涉及文档：{loc}"
    modify_docs = corrected_point.get("modify_docs")
    if isinstance(modify_docs, list) and modify_docs:
        content += f"\n需修改的文档：{', '.join(modify_docs)}"

    safe_fn = (file_name or "doc").replace("\\", "/").split("/")[-1][:120]
    logical_name = f"user_fb_{feedback_kind}__{safe_fn}"

    doc = Document(
        page_content=content,
        metadata={
            "kb_entry_class": "user_audit_feedback",
            "feedback_kind": feedback_kind,
            "type": "audit_user_feedback",
            "origin_file_name": file_name or "",
            "collection_tag": collection or "",
        },
    )
    agent = ReviewAgent(collection or "regulations")
    agent.checkpoint_feedback_kb.add_documents(
        [doc],
        file_name=logical_name,
        category="audit_user_feedback",
    )


def feed_correction_to_kb(collection: str, file_name: str, corrected_point: dict) -> None:
    """将误报/弃用/修订写入独立「用户审核反馈」向量库。"""
    if getattr(settings, "async_correction_kb_feed", True):
        threading.Thread(
            target=feed_correction_to_kb_impl,
            args=(collection, file_name, corrected_point),
            daemon=True,
        ).start()
        return
    feed_correction_to_kb_impl(collection, file_name, corrected_point)


def apply_audit_point_correction(
    root_report: Dict[str, Any],
    point_index: int,
    *,
    correction_kind: CorrectionKind = "revision",
    sub_report_index: int = 0,
    feed_to_kb: bool = True,
    description: Optional[str] = None,
    suggestion: Optional[str] = None,
    action: Optional[str] = None,
    modify_docs: Optional[List[str]] = None,
    severity: Optional[str] = None,
    regulation_ref: Optional[str] = None,
    location: Optional[str] = None,
    category: Optional[str] = None,
    false_positive_reason: Optional[str] = None,
    deprecation_note: Optional[str] = None,
    replacement_point: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    在内存中应用纠正并返回 (target_report, corrected_for_log, inserted_point)。
    不访问数据库。
    """
    target = get_target_report_for_points(root_report, sub_report_index)
    points = target.get("audit_points") or []
    if point_index < 0 or point_index >= len(points):
        raise ValueError("point index out of range")

    point = points[point_index]
    original_snap = dict(point)
    inserted: Optional[Dict[str, Any]] = None

    if correction_kind == "false_positive":
        reason = (false_positive_reason or "").strip()
        if not reason:
            raise ValueError("false_positive_reason is required")
        corrected = dict(point)
        corrected["correction_kind"] = "false_positive"
        corrected["false_positive_reason"] = reason
        corrected["action"] = "无需修改"
        old_sug = (point.get("suggestion") or "").strip()
        corrected["suggestion"] = old_sug + (f"\n\n【误报说明】{reason}" if reason else "")
        points[point_index] = corrected
        corrected_for_log = corrected
    elif correction_kind == "deprecated":
        corrected = dict(point)
        corrected["deprecated"] = True
        corrected["correction_kind"] = "deprecated"
        corrected["deprecation_note"] = (deprecation_note or "").strip()
        corrected["action"] = "无需修改"
        corrected_for_log = dict(corrected)
        if replacement_point:
            desc = (replacement_point.get("description") or "").strip()
            if not desc:
                raise ValueError("replacement_point.description is required")
            inserted = {
                "category": (replacement_point.get("category") or "").strip() or "一致性",
                "location": (replacement_point.get("location") or "").strip(),
                "description": desc,
                "regulation_ref": (replacement_point.get("regulation_ref") or "").strip(),
                "suggestion": (replacement_point.get("suggestion") or "").strip(),
                "severity": (replacement_point.get("severity") or "low").strip() or "low",
                "modify_docs": replacement_point.get("modify_docs") if isinstance(replacement_point.get("modify_docs"), list) else [],
                "action": (replacement_point.get("action") or "立即修改").strip() or "立即修改",
                "correction_kind": "user_replacement",
                "replaces_deprecated_index": point_index,
            }
            points[point_index] = corrected
            points.insert(point_index + 1, inserted)
            corrected_for_log["replacement_point_added"] = inserted
        else:
            points[point_index] = corrected
    else:
        corrected = dict(point)
        apply_point_field_updates(
            corrected,
            description=description,
            suggestion=suggestion,
            action=action,
            modify_docs=modify_docs,
            severity=severity,
            regulation_ref=regulation_ref,
            location=location,
            category=category,
        )
        corrected.pop("correction_kind", None)
        corrected.pop("false_positive_reason", None)
        corrected.pop("deprecated", None)
        corrected.pop("deprecation_note", None)
        points[point_index] = corrected
        corrected_for_log = corrected

    recount_severity(target)
    if root_report.get("batch") and root_report.get("reports"):
        aggregate_batch_report_totals(root_report)

    return target, corrected_for_log, inserted


def persist_audit_point_correction(
    report_id: int,
    point_index: int,
    *,
    collection: str,
    file_name: str,
    root_report: Dict[str, Any],
    original_snap: Dict[str, Any],
    corrected_for_log: Dict[str, Any],
    feed_to_kb: bool = True,
) -> None:
    """写回报告 JSON、audit_corrections 表、可选向量库与操作日志。"""
    update_audit_report(report_id, root_report)
    save_audit_correction(
        report_id=report_id,
        point_index=point_index,
        collection=collection or "",
        file_name=file_name or "",
        original=original_snap,
        corrected=corrected_for_log,
        fed_to_kb=feed_to_kb,
    )
    if feed_to_kb:
        try:
            feed_correction_to_kb(collection, file_name, corrected_for_log)
        except Exception:
            _logger.exception("写入审核反馈向量库失败 report_id=%s point=%s", report_id, point_index)
    add_operation_log(
        op_type=OP_TYPE_CORRECTION,
        collection=collection or "",
        file_name=file_name or "",
        extra={
            "report_id": report_id,
            "point_index": point_index,
            "fed_to_kb": feed_to_kb,
            "corrected": corrected_for_log,
        },
        model_info=get_current_model_info(),
    )
