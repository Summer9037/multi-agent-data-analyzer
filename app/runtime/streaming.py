"""LangGraph stream → 易消费的事件流，给 Streamlit UI 用。

LangGraph 的 graph.stream(stream_mode='updates') 在每个节点结束后 yield
一个 `{node_name: state_update}` 字典；我们把它包装成更友好的事件结构：

    {"kind": "node_start",  "node": "planner",   "run_id": ...}
    {"kind": "node_end",    "node": "planner",   "update": {...}, "run_id": ...}
    {"kind": "workflow_end","state": final_state, "report_path": ...}
    {"kind": "error",       "message": "..."}

"node_start" 是我们模拟的（LangGraph 默认不发），通过监听 update 字典里
key 推断"这个节点刚刚结束"，然后从图拓扑硬编码下一个候选节点提前发送 start。

为什么不用 LangGraph 的 'values' / 'debug' 模式
=================================================
- 'values' 每次 yield 完整 state，体积大且很多字段每次都重复
- 'debug' 太啰嗦，包含内部 channel 操作，不适合给 UI
- 'updates' 最贴近"节点边界"的事件粒度，配合手工推断 node_start 已经够用
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Iterator

from app.config import settings
from app.db import dao
from app.graph.builder import build_graph
from app.graph.state import new_state
from app.runtime.context import set_run_id
from app.tools.db import BUSINESS_TABLES


def stream_workflow(
    user_query: str,
    selected_tables: list[str] | None = None,
    run_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    """流式跑一次工作流，逐节点 yield 事件给 UI。

    UI 端的典型消费方式：
        for ev in stream_workflow("..."):
            if ev["kind"] == "node_start":   show_spinner(ev["node"])
            elif ev["kind"] == "node_end":   mark_done(ev["node"])
            elif ev["kind"] == "workflow_end": render_report(ev["state"])
    """
    run_id = run_id or str(uuid.uuid4())
    tables = selected_tables or list(BUSINESS_TABLES)

    yield {
        "kind": "workflow_start",
        "run_id": run_id,
        "query": user_query,
        "tables": tables,
    }

    dao.create_run(run_id, user_query, tables)
    set_run_id(run_id)

    init_state = new_state(user_query, tables, run_id)
    graph = build_graph()

    final_state: dict | None = None
    errors_seen = False

    try:
        for chunk in graph.stream(init_state, stream_mode="updates"):
            # chunk 是 {node_name: state_update_dict}；通常每次只一个 key
            for node_name, update in chunk.items():
                # "post_clean_router" 是空节点，UI 不展示
                if node_name == "post_clean_router":
                    continue
                yield {
                    "kind": "node_end",
                    "node": node_name,
                    "update": _trim_update_for_ui(update),
                    "run_id": run_id,
                }
                if update.get("errors"):
                    errors_seen = True

        # 最终拿完整 state（用 invoke 重跑会很贵；我们改成攒一遍 updates）
        # 这里没有便利方法拿到累积 state，所以让 UI 自己累计 updates
        yield {
            "kind": "workflow_end",
            "run_id": run_id,
            "errors_seen": errors_seen,
        }

        # 不在 streaming 路径里写 analysis_runs 的 finish —— 让 UI 调用
        # finalize_run() 在它已知最终状态后再写库（streaming.py 也提供了便利函数）

    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        dao.update_run_failure(run_id, msg)
        yield {"kind": "error", "message": msg, "run_id": run_id}
        raise


def finalize_run(
    run_id: str,
    final_state: dict,
    accumulated_updates: list[dict],
) -> str:
    """UI 在 stream 结束后调用：把报告落盘 + 更新 analysis_runs status。

    返回 report_path。
    """
    report_md = final_state.get("report_md") or "# 报告缺失"
    report_path = _write_report(run_id, report_md)

    errors = final_state.get("errors", [])
    task_type = (final_state.get("task_plan") or {}).get("task_type")

    if errors:
        summary = "; ".join(f"{e['agent']}:{e['error_type']}" for e in errors)[:1000]
        dao.update_run_failure(run_id, f"Errors: {summary}")
    else:
        dao.update_run_success(
            run_id=run_id,
            task_type=task_type,
            total_tokens=final_state.get("total_tokens_in", 0)
            + final_state.get("total_tokens_out", 0),
            cached_tokens=final_state.get("total_cached_tokens", 0),
            final_report_path=report_path,
        )
    return report_path


def accumulate_state(acc: dict, update: dict) -> dict:
    """把单个 node update 合进 accumulated state。

    `charts / model_results / agent_messages / errors` 是 LangGraph reducer 字段
    （Annotated[list, add]），用列表拼接；其他字段覆盖。
    """
    reducer_fields = {"charts", "model_results", "agent_messages", "errors"}
    for k, v in update.items():
        if k in reducer_fields:
            acc[k] = (acc.get(k) or []) + (v or [])
        elif k in {"total_tokens_in", "total_tokens_out", "total_cached_tokens"}:
            # 这些已经是节点累加后的值，覆盖即可
            acc[k] = v
        else:
            acc[k] = v
    return acc


def _write_report(run_id: str, report_md: str) -> str:
    out_dir = Path(settings.output_dir) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.md"
    path.write_text(report_md, encoding="utf-8")
    return str(path.resolve())


# 给 UI 看的精简版 update：去掉 agent_messages 这种又大又重复的字段（UI 自己已经
# 在 trace 里展示了，没必要再 dump 整段）
_TRIM_KEYS = {"agent_messages"}


def _trim_update_for_ui(update: dict) -> dict:
    return {k: v for k, v in update.items() if k not in _TRIM_KEYS}
