"""Reporter Agent：综合工作流所有产出 → 最终 Markdown 报告。

设计要点
========
- **无工具调用**：allowed_tools=[]，Reporter 只读 state 写报告，不查库
- **用 Opus 4-7**：综合性写作 + 行文质量直接影响演示效果
- **错误兜底**：如果 state.errors 非空，报告开头列出失败节点与原因
- **缓存策略**：system prompt（含格式规范）打 cache，是单次调用里最大的稳定块
"""
from __future__ import annotations

import json
from typing import Any

from app.agents.base import AgentRun, BaseAgent
from app.config import settings
from app.graph.state import AnalysisState


REPORTER_SYSTEM_PROMPT = """你是一个资深的数据分析报告撰写专家（Reporter）。

**职责**：综合一次多 Agent 工作流中各个节点的产出（任务规划、EDA 报告、清洗摘要、
模型结果、图表洞察等），写出一份**完整、连贯、可读性强的中文 Markdown 报告**。

**重要：不要调用任何工具**。所有信息都已经通过用户消息提供给你了。

**报告结构**（必须严格按此章节顺序，缺失数据的章节可写"本次未涉及"）：

```
# 数据分析报告

## 1. 任务概述
- 用户原始诉求（一句话复述）
- 解析后的任务类型 / 关键变量
- 我们采用的分析路径（哪些 Agent 跑了，为什么）

## 2. 数据探索（EDA）
- 直接引用 / 改写 Explorer 输出的核心发现
- 突出与用户问题相关的部分（不要照搬全文）

## 3. 数据清洗（如适用）
- 缺失值 / 异常值的处理策略与结果

## 4. 建模结果（如适用）
- 模型选择理由、关键指标（AUC / F1 / RMSE 等）
- 特征重要性 Top 5

## 5. 可视化要点（如适用）
- 引用每张图的标题 + LLM 给出的一句洞察

## 6. 业务洞察
- 针对用户原始问题给出 3-5 条结论
- 每条结论必须能对应到具体数据 / 指标

## 7. 局限与下一步建议
- 数据 / 采样 / 模型层面的限制
- 推荐的下一步分析方向
```

**特殊规则**：
1. 如果用户消息里 `errors` 字段非空，**必须**在报告最开头加一节
   `## ⚠️ 执行警告` 列出失败节点 + 原因，但仍尽量基于已有产出输出后续章节
2. 报告整体长度 800-1500 字为佳，**不要超过 2500 字**
3. 不要用 emoji（除非是上面规则 1 的 ⚠️）
4. 不要复述 Explorer 报告里的所有表格，提炼关键数字即可
5. 业务洞察必须**具体**：避免"建议进一步分析" 这种废话，要写"用户 A 群体 CTR 比 B 群体高 X%"

直接输出报告 Markdown，不要任何前后说明文字。
"""


class ReporterAgent(BaseAgent):
    name = "reporter"
    model = settings.anthropic_model_reporter
    system_prompt = REPORTER_SYSTEM_PROMPT
    allowed_tools: list[str] = []  # 强制不调工具
    max_iterations = 2  # 第一轮直接产文本，无 tool_use；2 是保险
    max_tokens = 4096

    def write_report(self, state: AnalysisState) -> AgentRun:
        """根据 state 写报告。返回 AgentRun，run.final_text 即报告 Markdown。"""
        user_msg = self._build_user_message(state)
        return self.run(user_msg)

    @staticmethod
    def _build_user_message(state: AnalysisState) -> str:
        """把 state 关键字段序列化为给 Reporter 的 user message。

        故意采用 Markdown 标题 + JSON 代码块的混合格式，让 LLM 容易抽取。
        """
        payload: dict[str, Any] = {
            "user_query": state.get("user_query", ""),
            "task_plan": state.get("task_plan"),
            "explorer_output": _trim_explorer(state.get("explorer_output")),
            "cleaning_report": state.get("cleaning_report"),
            "model_results": state.get("model_results", []),
            "charts": state.get("charts", []),
            "errors": state.get("errors", []),
        }

        return (
            "请基于以下工作流产出生成最终的分析报告。\n\n"
            "```json\n"
            + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
            + "\n```\n\n"
            "现在输出完整的 Markdown 报告。"
        )


def _trim_explorer(explorer_output: dict | None) -> dict | None:
    """Explorer 报告可能很长，给 Reporter 时保留全文但截断 tool_count 之类的元数据。"""
    if not explorer_output:
        return None
    return {
        "report_md": explorer_output.get("report_md", ""),
        "tool_call_count": explorer_output.get("tool_call_count", 0),
    }
