"""Streamlit 侧边栏：状态指示 / 数据集选择 / Token 用量 / 历史 run。"""
from __future__ import annotations

import streamlit as st
from sqlalchemy import text

from app.config import settings
from app.tools.db import BUSINESS_TABLES, get_engine


def render_sidebar() -> dict:
    """渲染侧边栏，返回用户在侧边栏的选择（如 selected_tables）。"""
    with st.sidebar:
        st.markdown("### ⚙️ 配置")

        api_ok = bool(settings.anthropic_api_key) and not settings.anthropic_api_key.startswith("sk-ant-xxx")
        base_url = settings.anthropic_base_url or "(default)"
        st.markdown(
            f"- Anthropic API key: {'✅ Configured' if api_ok else '❌ Missing'}\n"
            f"- Base URL: `{base_url}`\n"
            f"- Planner / Reporter: `{settings.anthropic_model_planner}`\n"
            f"- Worker: `{settings.anthropic_model_worker}`"
        )

        db_ok, db_msg = _check_db()
        st.markdown(f"- Database: {'✅ ' + db_msg if db_ok else '❌ ' + db_msg}")

        st.markdown("---")
        st.markdown("### 📊 数据集")
        selected = []
        for t in BUSINESS_TABLES:
            if st.checkbox(t, value=True, key=f"tbl_{t}"):
                selected.append(t)

        st.markdown("---")
        st.markdown("### 💰 Token 用量（当前会话）")
        usage = st.session_state.get("token_usage", {})
        if usage:
            total_in = usage.get("total_input", 0)
            cached = usage.get("cached", 0)
            output = usage.get("output", 0)
            hit = (cached / total_in * 100) if total_in > 0 else 0.0
            st.markdown(
                f"- total_input: `{total_in:,}`\n"
                f"- cached_input: `{cached:,}`\n"
                f"- output: `{output:,}`\n"
                f"- **cache_hit: `{hit:.1f}%`**"
            )
        else:
            st.caption("尚无运行")

        st.markdown("---")
        st.markdown("### 🗂️ 历史 Run（最近 10 条）")
        _render_history()

    return {"selected_tables": selected}


def _check_db() -> tuple[bool, str]:
    try:
        with get_engine().connect() as c:
            n = c.execute(text("SELECT COUNT(*) FROM raw_sample")).scalar_one()
        return True, f"Connected ({int(n):,} rows in raw_sample)"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:80]}"


def _render_history() -> None:
    try:
        with get_engine().connect() as c:
            rows = c.execute(
                text(
                    "SELECT run_id, status, task_type, user_query, "
                    "started_at FROM analysis_runs ORDER BY started_at DESC LIMIT 10"
                )
            ).fetchall()
    except Exception as e:
        st.caption(f"(load failed: {e})")
        return

    if not rows:
        st.caption("尚无历史")
        return

    for r in rows:
        emoji = "✅" if r.status == "success" else ("❌" if r.status == "failed" else "⏳")
        task = r.task_type or "?"
        short_q = (r.user_query[:32] + "…") if len(r.user_query) > 32 else r.user_query
        st.markdown(
            f"<small>{emoji} <code>{task}</code> {short_q}</small>",
            unsafe_allow_html=True,
        )
