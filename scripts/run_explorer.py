"""命令行入口：跑一次 Explorer Agent，验证 M2 端到端。

用法：
    python scripts/run_explorer.py "分析女性用户在不同年龄层的广告点击行为"
    python scripts/run_explorer.py "整体了解三张表的结构和分布"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.explorer import ExplorerAgent


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/run_explorer.py "<你的分析需求>"')
        sys.exit(1)
    query = " ".join(sys.argv[1:])

    print(f"\n=== Explorer Agent ===")
    print(f"Model: {ExplorerAgent.model}")
    print(f"Query: {query}\n")
    print("Running...")

    agent = ExplorerAgent()
    run = agent.run(query)

    print("\n=== Steps trace ===")
    for i, step in enumerate(run.steps):
        if step.role == "tool_call":
            print(f"[{i:2d}] CALL    {step.content[:160]}")
        elif step.role == "tool_result":
            preview = step.content.replace("\n", " ")[:160]
            print(f"[{i:2d}] RESULT  ({step.duration_ms}ms) {preview}")
        elif step.role == "thought":
            preview = step.content.replace("\n", " ")[:160]
            print(f"[{i:2d}] THINK   {preview}")
        elif step.role == "final":
            print(f"[{i:2d}] FINAL   (full text below)")

    print("\n=== Final Report ===")
    print(run.final_text)

    print("\n=== Token Usage ===")
    print(f"  uncached_input: {run.total_tokens_in:>7,}")
    print(f"  cached_input:   {run.total_cached_tokens:>7,}")
    print(f"  total_input:    {run.total_tokens_in + run.total_cached_tokens:>7,}")
    print(f"  output_tokens:  {run.total_tokens_out:>7,}")
    total_in = run.total_tokens_in + run.total_cached_tokens
    if total_in > 0:
        cache_pct = run.total_cached_tokens / total_in * 100
        print(f"  cache_hit:      {cache_pct:5.1f}%")
    print(f"  stop_reason:    {run.stopped_reason}")


if __name__ == "__main__":
    main()
