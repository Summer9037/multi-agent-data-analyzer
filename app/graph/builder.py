"""LangGraph StateGraph 构建器（M4 完整版）。

完整工作流拓扑
==============

    START
      ↓
    planner
      ↓
    explorer
      ↓
    [conditional: should_clean]
      ├──"cleaner"────→ cleaner ──┐
      └──"skip_cleaning"──────────┤
                                  ↓
                          [conditional: should_model]
                                  ├──"modeler"────→ modeler ──┐
                                  └──"skip_modeling"──────────┤
                                                              ↓
                                                          visualizer
                                                              ↓
                                                          reporter
                                                              ↓
                                                             END

关键设计
========
1. **两个 conditional_edges**：should_clean 在 explorer 后，should_model 在 cleaner/skip 之后
2. **post_clean 路由节点**：当 skip_cleaning 时不能直接连到 should_model 的判断里
   （LangGraph 要求 conditional 必须挂在节点上，不能挂在虚拟交叉点上），
   所以加一个空 op 的 "post_clean_router" 节点作为汇聚点
3. **visualizer 在每条路径上都走**：哪怕是 eda_only，画 3-5 张图也有助于 Reporter
4. 每个节点都做 try/except 兜底：节点内部失败时把错误塞 state.errors，
   图正常继续向下走（让 Reporter 在报告开头列错误）
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.cleaner import CleanerAgent
from app.agents.explorer import ExplorerAgent
from app.agents.modeler import ModelerAgent
from app.agents.planner import PlannerAgent
from app.agents.reporter import ReporterAgent
from app.agents.visualizer import VisualizerAgent
from app.db import dao
from app.graph.routing import should_clean, should_model
from app.graph.state import (
    AgentMessage,
    AnalysisState,
    ErrorEntry,
    ExplorerOutput,
)


# ============ 通用工具 ============

def _agent_run_to_messages(agent_name: str, run) -> list[AgentMessage]:
    out: list[AgentMessage] = []
    for s in run.steps:
        out.append(
            AgentMessage(
                agent=agent_name,
                role=s.role,
                content=s.content,
                tool_name=s.tool_name,
                duration_ms=int(s.duration_ms or 0),
                tokens_in=int(s.tokens_in or 0),
                tokens_out=int(s.tokens_out or 0),
                cached_tokens=int(s.cached_tokens or 0),
            )
        )
    return out


def _token_delta(state: AnalysisState, run) -> dict[str, int]:
    """汇总 token 增量字段，供节点返回 dict 用。"""
    return {
        "total_tokens_in": state.get("total_tokens_in", 0) + run.total_tokens_in,
        "total_tokens_out": state.get("total_tokens_out", 0) + run.total_tokens_out,
        "total_cached_tokens": state.get("total_cached_tokens", 0)
        + run.total_cached_tokens,
    }


def _err(state: AnalysisState, node: str, e: Exception) -> dict[str, Any]:
    """节点异常时统一兜底：仅记录 errors，让图继续往下走。"""
    return {
        "current_node": node,
        "errors": [
            ErrorEntry(agent=node, error_type=type(e).__name__, message=str(e))
        ],
    }


# ============ 节点实现 ============

def planner_node(state: AnalysisState) -> dict[str, Any]:
    try:
        agent = PlannerAgent()
        plan, meta, run = agent.plan(state["user_query"])
        dao.log_agent_run(state["run_id"], run)

        update: dict[str, Any] = {
            "current_node": "planner",
            "task_plan": plan,
            "dataset_meta": meta,
            "agent_messages": _agent_run_to_messages("planner", run),
            **_token_delta(state, run),
        }
        if plan is None:
            update["errors"] = [
                ErrorEntry(
                    agent="planner",
                    error_type="MissingTaskPlan",
                    message="Planner 没有调用 submit_task_plan",
                )
            ]
        return update
    except Exception as e:
        return _err(state, "planner", e)


def explorer_node(state: AnalysisState) -> dict[str, Any]:
    try:
        plan = state.get("task_plan")
        if plan:
            sub = "\n".join(f"- {s}" for s in plan.get("subtasks", []))
            user_msg = (
                f"用户原始需求：{state['user_query']}\n\n"
                f"Planner 拆解的子任务：\n{sub}\n\n"
                f"请围绕上述子任务做数据探索。"
            )
        else:
            user_msg = state["user_query"]

        agent = ExplorerAgent()
        run = agent.run(user_msg)
        dao.log_agent_run(state["run_id"], run)

        explorer_output = ExplorerOutput(
            report_md=run.final_text,
            tool_call_count=sum(1 for s in run.steps if s.role == "tool_call"),
            key_findings=[],
        )
        return {
            "current_node": "explorer",
            "explorer_output": explorer_output,
            "agent_messages": _agent_run_to_messages("explorer", run),
            **_token_delta(state, run),
        }
    except Exception as e:
        return _err(state, "explorer", e)


def cleaner_node(state: AnalysisState) -> dict[str, Any]:
    try:
        plan = state.get("task_plan") or {}
        # 选源表：优先 Planner 标注的 relevant_tables 里第一个；缺省 user_profile
        relevant = plan.get("relevant_tables") or []
        source_table = "user_profile"
        for t in relevant:
            if t in {"user_profile", "ad_feature", "raw_sample"}:
                # 优先选缺失最严重的 user_profile
                if t == "user_profile":
                    source_table = "user_profile"
                    break
                source_table = t

        explorer_md = (state.get("explorer_output") or {}).get("report_md", "")

        agent = CleanerAgent()
        report, run = agent.clean(state["user_query"], source_table, explorer_md)
        dao.log_agent_run(state["run_id"], run)

        update: dict[str, Any] = {
            "current_node": "cleaner",
            "cleaning_report": report,
            "agent_messages": _agent_run_to_messages("cleaner", run),
            **_token_delta(state, run),
        }
        if report is None:
            update["errors"] = [
                ErrorEntry(
                    agent="cleaner",
                    error_type="MissingCleaningReport",
                    message="Cleaner 没有调用 submit_cleaning_report",
                )
            ]
        return update
    except Exception as e:
        return _err(state, "cleaner", e)


def modeler_node(state: AnalysisState) -> dict[str, Any]:
    try:
        plan = state.get("task_plan") or {}
        has_cleaned = state.get("cleaning_report") is not None

        agent = ModelerAgent()
        results, run = agent.model_train(
            state["user_query"],
            target_column=plan.get("target_column"),
            has_cleaned=has_cleaned,
        )
        dao.log_agent_run(state["run_id"], run)

        return {
            "current_node": "modeler",
            "model_results": results,  # reducer 追加到 list
            "agent_messages": _agent_run_to_messages("modeler", run),
            **_token_delta(state, run),
        }
    except Exception as e:
        return _err(state, "modeler", e)


def visualizer_node(state: AnalysisState) -> dict[str, Any]:
    try:
        plan = state.get("task_plan")
        # 如果 Modeler 跑过，可以告诉 Visualizer 宽表名（feature_wide_<run_id>）
        feature_table = None
        if state.get("model_results"):
            feature_table = f"feature_wide_{state['run_id'].replace('-', '_')}"

        agent = VisualizerAgent()
        charts, run = agent.visualize(
            state["user_query"],
            task_plan=dict(plan) if plan else None,
            feature_table=feature_table,
        )
        dao.log_agent_run(state["run_id"], run)

        return {
            "current_node": "visualizer",
            "charts": charts,  # reducer 追加
            "agent_messages": _agent_run_to_messages("visualizer", run),
            **_token_delta(state, run),
        }
    except Exception as e:
        return _err(state, "visualizer", e)


def reporter_node(state: AnalysisState) -> dict[str, Any]:
    try:
        agent = ReporterAgent()
        run = agent.write_report(state)
        dao.log_agent_run(state["run_id"], run)
        return {
            "current_node": "reporter",
            "report_md": run.final_text,
            "agent_messages": _agent_run_to_messages("reporter", run),
            **_token_delta(state, run),
        }
    except Exception as e:
        return {
            **_err(state, "reporter", e),
            "report_md": f"# 报告生成失败\n\n{type(e).__name__}: {e}",
        }


def post_clean_router(state: AnalysisState) -> dict[str, Any]:
    """空 op 节点：作为 cleaner / skip_cleaning 两条路径的汇聚点，
    让后续的 should_model conditional_edges 可以挂在它身上。"""
    return {"current_node": "post_clean_router"}


# ============ 图装配 ============

def build_graph():
    """构建完整的多 Agent 工作流图（含两层 conditional 路由）。"""
    g = StateGraph(AnalysisState)

    g.add_node("planner", planner_node)
    g.add_node("explorer", explorer_node)
    g.add_node("cleaner", cleaner_node)
    g.add_node("post_clean_router", post_clean_router)
    g.add_node("modeler", modeler_node)
    g.add_node("visualizer", visualizer_node)
    g.add_node("reporter", reporter_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "explorer")

    # 第一层条件路由：是否需要清洗
    g.add_conditional_edges(
        "explorer",
        should_clean,
        {
            "cleaner": "cleaner",
            "skip_cleaning": "post_clean_router",
        },
    )
    g.add_edge("cleaner", "post_clean_router")

    # 第二层条件路由：是否需要建模
    g.add_conditional_edges(
        "post_clean_router",
        should_model,
        {
            "modeler": "modeler",
            "skip_modeling": "visualizer",
        },
    )
    g.add_edge("modeler", "visualizer")

    g.add_edge("visualizer", "reporter")
    g.add_edge("reporter", END)

    return g.compile()
