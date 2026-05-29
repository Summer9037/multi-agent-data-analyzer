"""M3 冒烟测试：跑一次完整的 Planner → Explorer → Reporter 三节点工作流。

用法：
    python scripts/smoke_test.py
    python scripts/smoke_test.py --query "训练 CTR 预测模型，给出 top10 重要特征"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.runtime.orchestrator import run_workflow


DEFAULT_QUERY = "整体了解三张表的结构和分布"


def main() -> None:
    parser = argparse.ArgumentParser(description="M3 工作流冒烟测试")
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="自然语言分析需求",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("M3 Smoke Test: Planner -> Explorer -> Reporter")
    print("=" * 60)

    final_state = run_workflow(args.query)

    # === 概览输出 ===
    print("\n" + "=" * 60)
    print("Workflow Summary")
    print("=" * 60)

    plan = final_state.get("task_plan")
    if plan:
        print(f"\n[TaskPlan]")
        print(f"  task_type      : {plan.get('task_type')}")
        print(f"  target_column  : {plan.get('target_column')}")
        print(f"  need_cleaning  : {plan.get('need_cleaning')}")
        print(f"  relevant_tables: {plan.get('relevant_tables')}")
        print(f"  subtasks       : ")
        for s in plan.get("subtasks", []):
            print(f"    - {s}")
        print(f"  rationale      : {plan.get('rationale')}")
    else:
        print("\n[TaskPlan] MISSING (planner failed to call submit_task_plan)")

    explorer = final_state.get("explorer_output")
    if explorer:
        print(f"\n[Explorer]")
        print(f"  tool_calls     : {explorer.get('tool_call_count')}")
        print(f"  report_chars   : {len(explorer.get('report_md', ''))}")

    print(f"\n[Agent messages] total={len(final_state.get('agent_messages', []))}")
    by_agent: dict[str, int] = {}
    for m in final_state.get("agent_messages", []):
        by_agent[m["agent"]] = by_agent.get(m["agent"], 0) + 1
    for ag, n in by_agent.items():
        print(f"  {ag:<10}: {n} steps")

    print(f"\n[Tokens]")
    tin = final_state.get("total_tokens_in", 0)
    tout = final_state.get("total_tokens_out", 0)
    tcache = final_state.get("total_cached_tokens", 0)
    total = tin + tcache
    print(f"  uncached_input : {tin:>7,}")
    print(f"  cached_input   : {tcache:>7,}")
    print(f"  total_input    : {total:>7,}")
    print(f"  output         : {tout:>7,}")
    if total > 0:
        print(f"  cache_hit      : {tcache / total * 100:5.1f}%")

    errors = final_state.get("errors", [])
    if errors:
        print(f"\n[Errors]")
        for e in errors:
            print(f"  {e['agent']}: {e['error_type']}: {e['message'][:200]}")

    print(f"\n[Report path] {final_state.get('report_path')}")
    print("\n--- Final report preview (first 800 chars) ---")
    print((final_state.get("report_md") or "")[:800])
    print("--- end ---")


if __name__ == "__main__":
    main()
