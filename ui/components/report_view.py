"""最终 Markdown 报告的渲染 + 下载按钮。"""
from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_report(report_md: str | None, report_path: str | None = None) -> None:
    """渲染最终报告，并提供 .md 下载按钮。"""
    if not report_md:
        st.info("尚未生成报告。")
        return

    if report_path:
        try:
            md_bytes = Path(report_path).read_bytes()
            fname = Path(report_path).name
            st.download_button(
                "⬇️ 下载报告 (.md)",
                data=md_bytes,
                file_name=fname,
                mime="text/markdown",
            )
        except Exception:
            pass

    st.markdown(report_md, unsafe_allow_html=False)
