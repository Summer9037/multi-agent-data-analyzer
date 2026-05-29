"""run_workflow：一次端到端的多 Agent 分析。

职责
====
1. 生成 run_id，把"运行开始"记录到 analysis_runs
2. 调 LangGraph 跑全流程
3. 把 Reporter 产出的 Markdown 报告落盘到 outputs/reports/<run_id>.md
4. 把"运行结束"（success / failed）+ token 汇总更新回 analysis_runs
5. 返回最终 AnalysisState 给调用方（CLI / Streamlit）
"""
from __future__ import annotations

import uuid
from pathlib import Path

from app.config import settings
from app.db import dao
from app.graph.builder import build_graph
from app.graph.state import AnalysisState, new_state
from app.runtime.context import set_run_id
from app.tools.db import BUSINESS_TABLES


def run_workflow(
    user_query: str,
    selected_tables: list[str] | None = None,
    run_id: str | None = None,
) -> AnalysisState:
    """跑一次完整工作流，落盘报告 + 写库 + 返回最终 state。"""
    run_id = run_id or str(uuid.uuid4())
    tables = selected_tables or list(BUSINESS_TABLES)

    print(f"[orchestrator] run_id={run_id}")
    print(f"[orchestrator] query={user_query!r}")
    print(f"[orchestrator] tables={tables}")

    dao.create_run(run_id, user_query, tables)
    set_run_id(run_id)

    init_state = new_state(user_query, tables, run_id)
    graph = build_graph()

    try:
        final_state: AnalysisState = graph.invoke(init_state)  # type: ignore[assignment]
    except Exception as e:
        # 整个图层面挂掉（一般是节点未捕获的异常 / 依赖问题）
        msg = f"{type(e).__name__}: {e}"
        print(f"[orchestrator] FATAL: {msg}")
        dao.update_run_failure(run_id, msg)
        raise

    # 落盘报告
    report_md = final_state.get("report_md") or "# 报告缺失"
    report_path = _write_report(run_id, report_md)
    final_state["report_path"] = report_path

    # 决定状态：errors 非空 → failed（仍存报告）；否则 → success
    errors = final_state.get("errors", [])
    task_type = (final_state.get("task_plan") or {}).get("task_type")

    if errors:
        err_summary = "; ".join(
            f"{e['agent']}:{e['error_type']}" for e in errors
        )[:1000]
        dao.update_run_failure(run_id, f"Errors in nodes: {err_summary}")
        print(f"[orchestrator] finished with errors: {err_summary}")
    else:
        dao.update_run_success(
            run_id=run_id,
            task_type=task_type,
            total_tokens=final_state.get("total_tokens_in", 0)
            + final_state.get("total_tokens_out", 0),
            cached_tokens=final_state.get("total_cached_tokens", 0),
            final_report_path=report_path,
        )
        print(f"[orchestrator] success. report -> {report_path}")

    return final_state


def _write_report(run_id: str, report_md: str) -> str:
    """把报告写到 outputs/reports/<run_id>.md，返回绝对路径字符串。"""
    out_dir = Path(settings.output_dir) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.md"
    path.write_text(report_md, encoding="utf-8")
    return str(path.resolve())
