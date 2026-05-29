"""快速查看最近的 analysis_runs 与运行时物理表。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.tools.db import get_engine


def main() -> None:
    with get_engine().connect() as c:
        print("=== 最近 5 个 runs ===")
        rows = c.execute(
            text(
                "SELECT run_id, status, task_type, total_tokens, "
                "cached_tokens, started_at "
                "FROM analysis_runs ORDER BY started_at DESC LIMIT 5"
            )
        ).fetchall()
        for r in rows:
            print(f"  {r.run_id}  {r.status:8s}  task={r.task_type}  "
                  f"tokens={r.total_tokens}  cached={r.cached_tokens}")

        print()
        print("=== 运行时物理表（cleaned_xxx / feature_wide_xxx） ===")
        rows = c.execute(
            text(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = 'public' "
                "AND (tablename LIKE 'cleaned_%' "
                "     OR tablename LIKE 'feature_wide_%' "
                "     OR tablename LIKE 'cleaning_work_%') "
                "ORDER BY tablename"
            )
        ).fetchall()
        for (name,) in rows:
            n = c.execute(text(f"SELECT COUNT(*) FROM {name}")).scalar_one()
            print(f"  {name}   rows={n:,}")

        print()
        print("=== 最新一个 run 的 agent 分布 ===")
        rows = c.execute(
            text(
                "SELECT agent_name, COUNT(*) AS steps, "
                "SUM(tokens_in) AS tin, SUM(tokens_out) AS tout "
                "FROM agent_execution_logs "
                "WHERE run_id = (SELECT run_id FROM analysis_runs "
                "                ORDER BY started_at DESC LIMIT 1) "
                "GROUP BY agent_name ORDER BY agent_name"
            )
        ).fetchall()
        for r in rows:
            print(f"  {r.agent_name:11s}  steps={r.steps:3d}  "
                  f"tokens_in={r.tin:>6}  tokens_out={r.tout:>6}")


if __name__ == "__main__":
    main()
