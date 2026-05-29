"""分析运行的数据访问层：写 analysis_runs / agent_execution_logs。

为什么独立成 DAO 而不是塞到 orchestrator
======================================
- orchestrator 只关心"图怎么跑"，不该关心"日志怎么入库"
- 写库逻辑集中后方便 M5 的 UI 直接查 PG 展示历史 run

所有函数都封装异常：DB 写失败不应该让整个工作流崩溃
（控制台打 warning，让分析结果文件仍然落盘）。
"""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text

from app.agents.base import AgentRun
from app.tools.db import get_engine


def create_run(run_id: str, user_query: str, selected_tables: list[str]) -> None:
    """工作流开始：插入一条 analysis_runs，status='running'。"""
    try:
        eng = get_engine()
        with eng.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO analysis_runs (run_id, user_query, status, selected_tables)
                    VALUES (CAST(:run_id AS UUID), :q, 'running', CAST(:tables AS JSONB))
                    """
                ),
                {
                    "run_id": run_id,
                    "q": user_query,
                    "tables": json.dumps(selected_tables),
                },
            )
    except Exception as e:
        print(f"[dao.create_run] WARN: {type(e).__name__}: {e}")


def update_run_success(
    run_id: str,
    task_type: str | None,
    total_tokens: int,
    cached_tokens: int,
    final_report_path: str | None,
) -> None:
    """工作流成功：填充 finished_at、status='success'、token 汇总。"""
    try:
        eng = get_engine()
        with eng.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE analysis_runs
                    SET status = 'success',
                        task_type = :task_type,
                        finished_at = NOW(),
                        total_tokens = :total_tokens,
                        cached_tokens = :cached_tokens,
                        final_report_path = :report_path
                    WHERE run_id = CAST(:run_id AS UUID)
                    """
                ),
                {
                    "run_id": run_id,
                    "task_type": task_type,
                    "total_tokens": int(total_tokens),
                    "cached_tokens": int(cached_tokens),
                    "report_path": final_report_path,
                },
            )
    except Exception as e:
        print(f"[dao.update_run_success] WARN: {type(e).__name__}: {e}")


def update_run_failure(run_id: str, error_message: str) -> None:
    """工作流失败：status='failed'，error_message 字段记录原因。"""
    try:
        eng = get_engine()
        with eng.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE analysis_runs
                    SET status = 'failed',
                        finished_at = NOW(),
                        error_message = :msg
                    WHERE run_id = CAST(:run_id AS UUID)
                    """
                ),
                {"run_id": run_id, "msg": error_message[:4000]},
            )
    except Exception as e:
        print(f"[dao.update_run_failure] WARN: {type(e).__name__}: {e}")


def log_agent_run(run_id: str, agent_run: AgentRun) -> None:
    """把一个 Agent 的所有 steps 批量写入 agent_execution_logs。

    step_index 在单个 agent 内 0..N 连续，跨 agent 不连续 —— 这样 SELECT
    的时候用 (agent_name, step_index) 排序就能还原每个 Agent 的执行流。
    """
    if not agent_run.steps:
        return
    rows: list[dict[str, Any]] = []
    for i, step in enumerate(agent_run.steps):
        rows.append(
            {
                "run_id": run_id,
                "agent_name": agent_run.agent_name,
                "step_index": i,
                "role": step.role,
                "tool_name": step.tool_name,
                "tool_args": (
                    json.dumps(step.tool_args, ensure_ascii=False, default=str)
                    if step.tool_args
                    else None
                ),
                "tool_result_summary": (
                    step.content[:2000] if step.role == "tool_result" else None
                ),
                "duration_ms": int(step.duration_ms or 0),
                "tokens_in": int(step.tokens_in or 0),
                "tokens_out": int(step.tokens_out or 0),
            }
        )

    try:
        eng = get_engine()
        with eng.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO agent_execution_logs
                      (run_id, agent_name, step_index, role, tool_name,
                       tool_args, tool_result_summary, duration_ms,
                       tokens_in, tokens_out)
                    VALUES
                      (CAST(:run_id AS UUID), :agent_name, :step_index, :role, :tool_name,
                       CAST(:tool_args AS JSONB), :tool_result_summary, :duration_ms,
                       :tokens_in, :tokens_out)
                    """
                ),
                rows,
            )
    except Exception as e:
        print(f"[dao.log_agent_run] WARN: {type(e).__name__}: {e}")
