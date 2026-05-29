"""运行时上下文：用 ContextVar 暴露当前 run_id。

为什么用 ContextVar 而不是把 run_id 塞到每个 tool 的入参里
============================================================
1. Tool 调用 schema 越小越好，让 LLM 关注业务参数；run_id 是系统级元数据
2. LLM 有概率忘传或编造 run_id；从上下文取，杜绝这种风险
3. 多个 Agent 共享同一 run_id，set 一次即可

Orchestrator 在 invoke graph 之前 `set_run_id(...)`，所有需要 run_id 的
tool 内部用 `get_run_id()` 拿到。
"""
from __future__ import annotations

import contextvars


_current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_run_id", default=None
)


def set_run_id(run_id: str) -> None:
    _current_run_id.set(run_id)


def get_run_id() -> str:
    rid = _current_run_id.get()
    if not rid:
        raise RuntimeError(
            "No run_id set in context. Did orchestrator forget to call set_run_id()?"
        )
    return rid


def get_run_id_for_sql() -> str:
    """返回适合拼进 SQL 标识符的 run_id 形式（UUID 的 '-' 替换为 '_'）。"""
    return get_run_id().replace("-", "_")
