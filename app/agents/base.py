"""BaseAgent：Claude tool_use 多轮循环 + prompt caching 的统一封装。

所有 Agent (Planner / Explorer / Cleaner / Modeler / Visualizer / Reporter)
继承这个基类。子类只需覆盖：
- name / model / system_prompt
- allowed_tools (None 表示全部)
- max_iterations / max_tokens
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.llm.cache import cache_tools, with_cache
from app.llm.client import get_client
from app.tools.registry import TOOL_SPECS, execute_tool


@dataclass
class AgentStep:
    """Agent 内部一次迭代记录（用于日志与 UI 流式渲染）。"""

    role: str  # "thought" | "tool_call" | "tool_result" | "final"
    content: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cached_tokens: int = 0
    duration_ms: int = 0


@dataclass
class AgentRun:
    """一次 Agent 完整运行的产出。"""

    agent_name: str
    final_text: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cached_tokens: int = 0
    stopped_reason: str = "ok"  # ok / max_iterations / error


class BaseAgent:
    """Claude tool_use 多轮循环 Agent 基类。"""

    name: str = "base"
    model: str = settings.anthropic_model_worker
    system_prompt: str = ""
    allowed_tools: list[str] | None = None
    max_iterations: int = settings.agent_max_iterations
    max_tokens: int = 4096

    # === 公开接口 ===
    def run(self, user_message: str, extra_context: str | None = None) -> AgentRun:
        """运行 Agent。返回 AgentRun 含 final_text + steps + token usage。

        extra_context: 可选的稳定上下文（如数据库 schema 描述），会单独
        打 cache_control 标记，跨多次调用复用缓存。
        """
        client = get_client()
        run = AgentRun(agent_name=self.name)

        messages: list[dict] = [
            {"role": "user", "content": self._build_initial_user(user_message, extra_context)}
        ]

        for _ in range(self.max_iterations):
            t0 = time.time()
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=with_cache(self.system_prompt),
                tools=cache_tools(self._select_tools()),
                messages=messages,
            )
            duration_ms = int((time.time() - t0) * 1000)

            self._accumulate_usage(run, response.usage)

            text_chunks, tool_uses, assistant_blocks = self._split_response(response)

            # 把 thought / final 记录下来
            if text_chunks:
                run.steps.append(
                    AgentStep(
                        role="thought" if response.stop_reason == "tool_use" else "final",
                        content="\n".join(text_chunks),
                        tokens_in=response.usage.input_tokens,
                        tokens_out=response.usage.output_tokens,
                        cached_tokens=getattr(
                            response.usage, "cache_read_input_tokens", 0
                        )
                        or 0,
                        duration_ms=duration_ms,
                    )
                )

            if response.stop_reason != "tool_use":
                run.final_text = "\n".join(text_chunks)
                return run

            messages.append({"role": "assistant", "content": assistant_blocks})

            # 执行所有 tool_use（一轮内可能并行多个）
            tool_results = []
            for tu in tool_uses:
                tool_results.append(self._execute_one_tool(tu, run))
            messages.append({"role": "user", "content": tool_results})

        run.stopped_reason = "max_iterations"
        run.final_text = "(达到最大迭代次数，未产出最终结果)"
        return run

    # === 子类可覆盖的钩子 ===
    def _select_tools(self) -> list[dict]:
        if self.allowed_tools is None:
            return list(TOOL_SPECS)
        return [t for t in TOOL_SPECS if t["name"] in self.allowed_tools]

    # === 私有辅助 ===
    @staticmethod
    def _build_initial_user(message: str, extra_context: str | None) -> list[dict]:
        blocks: list[dict] = []
        if extra_context:
            blocks.append(
                {
                    "type": "text",
                    "text": extra_context,
                    "cache_control": {"type": "ephemeral"},
                }
            )
        blocks.append({"type": "text", "text": message})
        return blocks

    @staticmethod
    def _accumulate_usage(run: AgentRun, usage: Any) -> None:
        run.total_tokens_in += usage.input_tokens
        run.total_tokens_out += usage.output_tokens
        run.total_cached_tokens += (
            getattr(usage, "cache_read_input_tokens", 0) or 0
        )

    @staticmethod
    def _split_response(response: Any) -> tuple[list[str], list[Any], list[dict]]:
        """把 response.content 拆成 (text 列表, tool_use 列表, assistant blocks)。"""
        text_chunks: list[str] = []
        tool_uses: list[Any] = []
        assistant_blocks: list[dict] = []
        for block in response.content:
            if block.type == "text":
                text_chunks.append(block.text)
                assistant_blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                tool_uses.append(block)
                assistant_blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
        return text_chunks, tool_uses, assistant_blocks

    def _execute_one_tool(self, tu: Any, run: AgentRun) -> dict:
        """执行一次 tool_use，记录到 run.steps，返回 tool_result block。"""
        run.steps.append(
            AgentStep(
                role="tool_call",
                content=f"{tu.name}({json.dumps(tu.input, ensure_ascii=False)})",
                tool_name=tu.name,
                tool_args=tu.input,
            )
        )
        t0 = time.time()
        result = execute_tool(tu.name, tu.input)
        duration_ms = int((time.time() - t0) * 1000)

        result_text = self._serialize_result(result)
        # tool_result 只在 step 里存一个截断摘要，给 UI 看
        summary = result_text[:500] + ("..." if len(result_text) > 500 else "")
        run.steps.append(
            AgentStep(
                role="tool_result",
                content=summary,
                tool_name=tu.name,
                duration_ms=duration_ms,
            )
        )
        return {
            "type": "tool_result",
            "tool_use_id": tu.id,
            "content": result_text,
        }

    @staticmethod
    def _serialize_result(result: dict, max_bytes: int = 8000) -> str:
        """序列化 tool 结果给 LLM。超过 max_bytes 自动截断标记 truncated。"""
        text = json.dumps(result, ensure_ascii=False, default=str)
        if len(text.encode("utf-8")) > max_bytes:
            text = text[:max_bytes] + ' ..."__truncated":true'
        return text
