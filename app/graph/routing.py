"""LangGraph conditional_edges 的路由函数。

约定
====
每个 should_X(state) 函数返回一个**字符串字面量**，对应
`graph.add_conditional_edges(node, should_X, {"key": "next_node"})`
里 mapping 的 key。

设计意图
========
把"走不走 Cleaner / 走不走 Modeler"这种决策**集中在路由层**，
而不是塞进每个 Agent 内部，让图的拓扑一眼看清楚：
- task_type == "eda_only"        → 跳过 Cleaner + Modeler
- task_type in {modeling/...}    → 走全链路
- need_cleaning == False         → 跳过 Cleaner（即使有建模也直接进 Modeler）

M3 范围
========
M3 只跑 Planner → Explorer → Reporter，不调用本文件里的路由。
但接口先稳定下来，M4 直接 import 用即可。
"""
from __future__ import annotations

from app.graph.state import AnalysisState


def should_clean(state: AnalysisState) -> str:
    """Explorer 之后：是否需要 Cleaner？

    返回:
      - "cleaner"        进 Cleaner 节点
      - "skip_cleaning"  跳过，直接到下一个判断点（should_model）
    """
    plan = state.get("task_plan")
    if not plan:
        return "skip_cleaning"
    if plan.get("task_type") == "eda_only":
        return "skip_cleaning"
    return "cleaner" if plan.get("need_cleaning", False) else "skip_cleaning"


def should_model(state: AnalysisState) -> str:
    """Cleaner（或 Explorer 直跳）之后：是否需要 Modeler？

    返回:
      - "modeler"        进 Modeler 节点
      - "skip_modeling"  跳过，直接进 Visualizer / Reporter
    """
    plan = state.get("task_plan")
    if not plan:
        return "skip_modeling"
    return (
        "modeler"
        if plan.get("task_type") in ("modeling", "comparison")
        else "skip_modeling"
    )
