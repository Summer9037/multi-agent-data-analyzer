"""多 Agent 自动化数据分析 — Streamlit 主入口。

启动：
    streamlit run ui/streamlit_app.py

布局
====
- 侧边栏：配置 / 数据集勾选 / Token 用量 / 历史 run
- 主区：
  - 自然语言 query 输入 + 示例按钮
  - "Run Analysis" 触发
  - Agent 执行轨迹（流式更新的卡片）
  - 完成后：Tab(报告 | 图表 | 任务计划 | 日志)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# 让 streamlit 子进程能 import app.*
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.runtime.streaming import (
    accumulate_state,
    finalize_run,
    stream_workflow,
)
from ui.components.agent_trace import TraceState, render_trace_panel
from ui.components.chart_grid import render_chart_grid
from ui.components.report_view import render_report
from ui.components.sidebar import render_sidebar


# ============ 页面配置 ============

st.set_page_config(
    page_title="多 Agent 数据分析",
    page_icon="🤖",
    layout="wide",
)


# ============ Session State 初始化 ============

def _init_state() -> None:
    if "trace_state" not in st.session_state:
        st.session_state.trace_state = TraceState()
    if "running" not in st.session_state:
        st.session_state.running = False
    if "accumulated" not in st.session_state:
        st.session_state.accumulated = {}
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "token_usage" not in st.session_state:
        st.session_state.token_usage = {}


_init_state()


# ============ 主区 ============

st.title("🤖 多 Agent 自动化数据分析系统")
st.caption(
    "自然语言 → Planner → Explorer → Cleaner → Modeler → Visualizer → Reporter，"
    "基于 LangGraph + Anthropic Claude tool_use 编排"
)

sidebar_config = render_sidebar()

# === Query 输入 ===

EXAMPLES = [
    "整体了解三张表的结构和分布",
    "分析女性用户在不同年龄层的广告点击行为",
    "训练 CTR 预测模型并给出 top10 重要特征",
    "对比男女用户在不同价位广告上的点击表现",
]

st.markdown("### 1️⃣ 输入你的分析需求")

cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    if cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
        st.session_state.last_query = ex

query = st.text_area(
    "自然语言分析需求",
    value=st.session_state.last_query,
    placeholder="比如：训练一个 CTR 预测模型并解释主要影响因素",
    height=80,
    label_visibility="collapsed",
)

run_clicked = st.button(
    "🚀 Run Analysis",
    type="primary",
    disabled=st.session_state.running or not query.strip(),
    use_container_width=False,
)

# === Trace 面板 ===

st.markdown("### 2️⃣ Agent 执行轨迹")
trace_placeholder = st.empty()


def _redraw_trace() -> None:
    """重画一次 trace 面板（在 stream 循环里反复调用）。"""
    with trace_placeholder.container():
        render_trace_panel(st.session_state.trace_state)


_redraw_trace()


# === 触发运行 ===

if run_clicked and query.strip():
    # 重置状态
    st.session_state.trace_state = TraceState()
    st.session_state.accumulated = {}
    st.session_state.running = True

    tables = sidebar_config["selected_tables"]
    if not tables:
        st.warning("请至少勾选一个数据表。")
        st.session_state.running = False
    else:
        try:
            for ev in stream_workflow(query.strip(), selected_tables=tables):
                kind = ev["kind"]
                if kind == "workflow_start":
                    st.session_state.trace_state.run_id = ev["run_id"]
                elif kind == "node_end":
                    node = ev["node"]
                    update = ev["update"]
                    st.session_state.trace_state.mark_done(node, update)
                    st.session_state.accumulated = accumulate_state(
                        st.session_state.accumulated, update
                    )
                    _redraw_trace()
                elif kind == "error":
                    st.error(f"工作流失败：{ev['message']}")
                elif kind == "workflow_end":
                    st.session_state.trace_state.workflow_done = True
                    st.session_state.trace_state.final_state = (
                        st.session_state.accumulated
                    )

            # 落盘 + 更新 analysis_runs
            run_id = st.session_state.trace_state.run_id
            if run_id:
                report_path = finalize_run(
                    run_id, st.session_state.accumulated, []
                )
                st.session_state.accumulated["report_path"] = report_path

            # 更新 token usage 给侧边栏
            tin = st.session_state.accumulated.get("total_tokens_in", 0)
            tcached = st.session_state.accumulated.get("total_cached_tokens", 0)
            tout = st.session_state.accumulated.get("total_tokens_out", 0)
            st.session_state.token_usage = {
                "total_input": tin + tcached,
                "cached": tcached,
                "output": tout,
            }

        except Exception as e:
            st.exception(e)
        finally:
            st.session_state.running = False

        _redraw_trace()
        st.success("✅ 工作流完成")
        st.rerun()


# === 完成后展示结果 ===

if (
    not st.session_state.running
    and st.session_state.trace_state.workflow_done
    and st.session_state.accumulated
):
    st.markdown("### 3️⃣ 分析结果")
    final = st.session_state.accumulated

    tab_report, tab_charts, tab_plan, tab_logs = st.tabs(
        ["📝 最终报告", "📊 图表", "🎯 任务计划", "🔍 节点日志"]
    )

    with tab_report:
        render_report(final.get("report_md"), final.get("report_path"))

    with tab_charts:
        render_chart_grid(final.get("charts") or [])

    with tab_plan:
        plan = final.get("task_plan") or {}
        if plan:
            st.markdown(
                f"- **task_type**: `{plan.get('task_type')}`\n"
                f"- **target_column**: `{plan.get('target_column')}`\n"
                f"- **need_cleaning**: `{plan.get('need_cleaning')}`\n"
                f"- **relevant_tables**: `{plan.get('relevant_tables')}`"
            )
            st.markdown("**Subtasks**:")
            for s in plan.get("subtasks", []):
                st.markdown(f"- {s}")
            st.markdown("**Rationale**:")
            st.caption(plan.get("rationale", ""))
        else:
            st.info("Planner 未产出 task_plan")

        cr = final.get("cleaning_report")
        if cr:
            st.markdown("---")
            st.markdown("**Cleaning Report**:")
            st.json(cr, expanded=False)

        mr = final.get("model_results") or []
        if mr:
            st.markdown("---")
            st.markdown("**Model Results**:")
            for m in mr:
                st.json(m, expanded=False)

    with tab_logs:
        from sqlalchemy import text

        from app.tools.db import get_engine

        run_id = st.session_state.trace_state.run_id
        if run_id:
            try:
                with get_engine().connect() as c:
                    rows = c.execute(
                        text(
                            "SELECT agent_name, step_index, role, tool_name, "
                            "duration_ms, tokens_in, tokens_out "
                            "FROM agent_execution_logs WHERE run_id = CAST(:rid AS UUID) "
                            "ORDER BY agent_name, step_index"
                        ),
                        {"rid": run_id},
                    ).fetchall()
                if rows:
                    import pandas as pd
                    df = pd.DataFrame(
                        rows,
                        columns=["agent", "step", "role", "tool", "ms", "in", "out"],
                    )
                    st.dataframe(df, use_container_width=True, height=400)
                else:
                    st.info("无日志记录")
            except Exception as e:
                st.error(f"日志查询失败: {e}")
