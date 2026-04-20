---
name: streamlit-ui-layout
description: Streamlit 页面布局安全（禁止 expander 嵌套、key 唯一、大 JSON 不落 session）。用户提到「Streamlit / expander 嵌套 / StreamlitAPIException / app.py 历史记录 UI」时使用。
---

# Streamlit UI 布局（本项目）

## 硬约束

- **`st.expander` 不得嵌套**：内层不要用第二个 `st.expander`，会抛 `StreamlitAPIException`。
- 本项目已在「历史生成记录」等场景踩坑：外层批次 expander + 内层按文件 expander → 必须改为 **selectbox/radio + 下方详情区**。

## 实现模式

1. **外层**保留「一条批次 / 一块功能区」一个 expander（若需要）。
2. **内层多文件/多子项**：`st.selectbox("选择文件…", options=range(n), format_func=...)`，根据选中索引渲染 `patch.report`、下载按钮等。
3. **大内容**：`st.code(json.dumps(...))` 截断展示 + 提供下载；不要把完整 `changes` 写入 DB 或 `st.session_state`。

## Widget key

- 循环渲染时 key 带 **`report_id` / `rec['id']` / 文件索引 / 文件名哈希`**，避免 rerun 后 key 冲突。

## 验证

- 改完 `src/app.py` 后本地打开对应页面，展开所有含子项的区块，确认无异常；必要时 `python -m py_compile src/app.py`。
