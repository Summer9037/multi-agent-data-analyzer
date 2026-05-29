"""Agent 执行轨迹的流式可视化（M5 核心亮点）。

特性
====
- 每个 Agent 一张卡片，颜色编码状态
  ⏳ pending(灰) / 🟦 running(蓝+spinner) / ✅ done(绿) / ❌ error(红)
- 卡片可展开看 tool 调用 JSON / 结果摘要 / token 用量 / 耗时
- 整个执行过程实时增量更新（用 st.empty() + LangGraph stream）
"""
from __future__ import annotations

import json
from typing import Any

import streamlit as st


# Agent 显示元信息：决定图标 + 中文显示名 + 描述
AGENT_META = {
    "planner": {"icon": "🎯", "label": "Planner", "desc": "解析需求 → 结构化 TaskPlan"},
    "explorer": {"icon": "🔍", "label": "Explorer", "desc": "数据探索 / EDA"},
    "cleaner": {"icon": "🧹", "label": "Cleaner", "desc": "缺失值 / 异常 / 编码"},
    "modeler": {"icon": "🤖", "label": "Modeler", "desc": "构建宽表 + 训练模型"},
    "visualizer": {"icon": "📊", "label": "Visualizer", "desc": "生成 Plotly 图表"},
    "reporter": {"icon": "📝", "label": "Reporter", "desc": "综合写最终报告"},
}

# 预期执行顺序（按 LangGraph 拓扑），用于占位
EXPECTED_ORDER = ["planner", "explorer", "cleaner", "modeler", "visualizer", "reporter"]


class TraceState:
    """累计 stream 事件 → 各 Agent 状态。

    用 st.session_state 持久化（跨 Streamlit re-run 保留）。
    """

    def __init__(self) -> None:
        # agent_name -> {status, update, error}
        self.agents: dict[str, dict[str, Any]] = {}
        self.workflow_done: bool = False
        self.final_state: dict | None = None
        self.run_id: str | None = None

    def mark_running(self, name: str) -> None:
        self.agents.setdefault(name, {"status": "running"})
        self.agents[name]["status"] = "running"

    def mark_done(self, name: str, update: dict) -> None:
        errors = update.get("errors") or []
        status = "error" if errors else "done"
        self.agents[name] = {"status": status, "update": update}


def render_trace_panel(state: TraceState) -> None:
    """渲染整个 trace 面板（在 st.container() 内调用）。

    每次 stream 收到事件后，应调用 panel.empty() 再 render_trace_panel(state)
    重新画一次（这是 Streamlit 流式更新的常规套路）。
    """
    # 决定要展示哪些 Agent：已经看到过的 + 后续可能的（按预期顺序）
    seen = list(self_seen_ordered(state.agents))
    remaining = [n for n in EXPECTED_ORDER if n not in state.agents]
    # 如果 task_plan 已经定了 task_type，可以从 remaining 里剔除被跳过的节点
    final_state = state.final_state or {}
    plan = final_state.get("task_plan") or {}
    if plan and plan.get("task_type") == "eda_only":
        remaining = [n for n in remaining if n not in {"cleaner", "modeler"}]
    elif plan and not plan.get("need_cleaning", True):
        remaining = [n for n in remaining if n != "cleaner"]

    all_to_show = seen + remaining

    cols = st.columns(min(3, max(1, len(all_to_show))))
    for i, name in enumerate(all_to_show):
        col = cols[i % len(cols)]
        with col:
            _render_card(name, state.agents.get(name))


def _render_card(name: str, slot: dict | None) -> None:
    """画单张 Agent 卡片。"""
    meta = AGENT_META.get(name, {"icon": "❓", "label": name, "desc": ""})
    status = (slot or {}).get("status", "pending")
    update = (slot or {}).get("update") or {}

    if status == "pending":
        bg, badge = "#f3f4f6", "⏳ Pending"
    elif status == "running":
        bg, badge = "#dbeafe", "🟦 Running"
    elif status == "done":
        bg, badge = "#d1fae5", "✅ Done"
    else:
        bg, badge = "#fee2e2", "❌ Error"

    st.markdown(
        f"""
        <div style="background:{bg};border-radius:8px;padding:10px 12px;margin-bottom:8px;">
          <div style="font-size:14px;font-weight:600;">{meta['icon']} {meta['label']}
            <span style="float:right;font-size:11px;color:#555;">{badge}</span>
          </div>
          <div style="font-size:11px;color:#666;margin-top:2px;">{meta['desc']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if slot:
        _render_card_details(name, slot)


def _render_card_details(name: str, slot: dict) -> None:
    """卡片下面的可展开详情（tool 调用 / 关键字段 / token）。"""
    update = slot.get("update") or {}
    with st.expander("展开详情", expanded=False):
        # 显示关键字段（按 Agent 类型不同选择展示）
        if name == "planner" and update.get("task_plan"):
            tp = update["task_plan"]
            st.markdown(
                f"- **task_type**: `{tp.get('task_type')}`\n"
                f"- **target_column**: `{tp.get('target_column')}`\n"
                f"- **need_cleaning**: `{tp.get('need_cleaning')}`"
            )
            if tp.get("subtasks"):
                st.markdown("**Subtasks**:")
                for s in tp["subtasks"]:
                    st.markdown(f"- {s}")
            if tp.get("rationale"):
                st.caption(f"💡 {tp['rationale']}")

        elif name == "explorer" and update.get("explorer_output"):
            eo = update["explorer_output"]
            st.markdown(
                f"- tool_calls: `{eo.get('tool_call_count')}`\n"
                f"- report 字符数: `{len(eo.get('report_md',''))}`"
            )

        elif name == "cleaner" and update.get("cleaning_report"):
            cr = update["cleaning_report"]
            st.markdown(
                f"- cleaned_table: `{cr.get('cleaned_table')}`\n"
                f"- rows: `{cr.get('rows_before')}` → `{cr.get('rows_after')}`"
            )
            if cr.get("operations"):
                st.markdown("**Operations**:")
                for op in cr["operations"]:
                    st.markdown(f"- `{op.get('op')}` — {op.get('detail')}")

        elif name == "modeler" and update.get("model_results"):
            for mr in update["model_results"]:
                st.markdown(f"**{mr.get('model_name')}**")
                m = mr.get("metrics") or {}
                st.markdown(
                    f"- AUC: `{m.get('auc')}`  F1: `{m.get('f1')}`  Acc: `{m.get('accuracy')}`"
                )
                fi = mr.get("feature_importance") or []
                if fi:
                    st.markdown("**Top features**:")
                    for f in fi[:10]:
                        st.markdown(f"- `{f.get('feature')}` ({f.get('importance')})")

        elif name == "visualizer" and update.get("charts"):
            for ch in update["charts"]:
                st.markdown(f"- **{ch.get('title') or ch.get('chart_id')}**")
                st.caption(f"💡 {ch.get('insight','')}")

        elif name == "reporter" and update.get("report_md"):
            st.caption(f"报告字符数: {len(update['report_md'])}")

        # token 用量增量（如果 update 里有 total_tokens_in 字段就显示）
        if "total_tokens_in" in update:
            st.markdown(
                "<small>累计 token (workflow 级): "
                f"in={update.get('total_tokens_in', 0):,} "
                f"cached={update.get('total_cached_tokens', 0):,} "
                f"out={update.get('total_tokens_out', 0):,}</small>",
                unsafe_allow_html=True,
            )

        if update.get("errors"):
            for e in update["errors"]:
                st.error(
                    f"{e.get('agent')}: {e.get('error_type')}: {e.get('message')}"
                )


def self_seen_ordered(agents: dict[str, dict]) -> list[str]:
    """已经看见过的 Agent 按预期顺序排序。"""
    return [n for n in EXPECTED_ORDER if n in agents]
