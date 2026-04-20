---
name: upload-display-filename
description: 规定上传审核场景下展示名与临时文件路径分离；报告与 modify_docs 须显示用户上传文件名。用户提到「tmp 文件名、临时文件、历史报告下拉、需修改文档显示错」时使用。
---

# 上传文件展示名 vs 临时路径

## 问题

Streamlit / 服务端常用 `NamedTemporaryFile` 落盘，基名形如 `tmp7zq2k8f.docx`。若该字符串进入 LLM 提示词或报告 JSON，界面会出现「需修改文档：tmp….docx」与历史列表中的临时名。

## 正确做法

1. **解析路径**：仅用于 `load_single_file(temp_path)`。
2. **展示名**：始终使用上传对象的 `name`（或解压后的 `archive/rel` 规则名），传入 `display_file_name` / `original_filename`。
3. **审核内核**：`review_file(..., display_file_name=展示名)`，内部用展示名填 prompt；用 `storage_basename=Path(temp).name` 做 `modify_docs` 中临时名替换。
4. **历史数据**：列表查询用 `effective_audit_report_display_name` + `sanitize_audit_report_dict` 兼容旧行。

## 代码锚点（本项目）

- `src/core/display_filename.py`：`effective_audit_report_display_name`、`sanitize_audit_report_dict`
- `src/core/reviewer.py`：`review_file`、`_rewrite_temp_modify_docs`
- `src/core/db.py`：`get_audit_report_file_names`、`get_audit_reports_by_file_name`
