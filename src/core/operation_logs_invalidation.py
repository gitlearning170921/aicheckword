"""使 Streamlit「操作记录」页 TTL 缓存失效；FastAPI 写入 operation_logs 后应调用。"""


def invalidate_operation_logs_cache() -> None:
    try:
        import streamlit as st

        st.session_state.pop("_ttl_cache__cached_operation_logs", None)
    except Exception:
        pass
