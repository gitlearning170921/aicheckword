"""兼容旧版 Streamlit API（如 rerun 名称变更）。"""

from __future__ import annotations

import streamlit as st


def streamlit_divider() -> None:
    """st.divider() 在 Streamlit 1.10+ 才有；1.9 等旧版用分隔线 markdown。"""
    div = getattr(st, "divider", None)
    if callable(div):
        div()
    else:
        st.markdown("---")


def streamlit_rerun() -> None:
    """优先 st.rerun()，旧版回退 experimental_rerun。"""
    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(fn):
        fn()
