"""图表网格：把 Visualizer 产出的 HTML 图嵌入 Streamlit。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components


def render_chart_grid(charts: list[dict[str, Any]]) -> None:
    """两列网格，每列一张图 + 标题 + 一句洞察。"""
    if not charts:
        st.info("本次运行未生成图表。")
        return

    cols = st.columns(2)
    for i, ch in enumerate(charts):
        col = cols[i % 2]
        with col:
            title = ch.get("title") or ch.get("chart_id") or "Chart"
            insight = ch.get("insight") or ""
            path = ch.get("path")
            st.markdown(f"**{title}**")
            if insight:
                st.caption(f"💡 {insight}")
            if path and Path(path).exists():
                try:
                    html = Path(path).read_text(encoding="utf-8")
                    components.html(html, height=420, scrolling=False)
                except Exception as e:
                    st.error(f"图表加载失败: {e}")
            else:
                st.warning(f"图表文件缺失: {path}")
