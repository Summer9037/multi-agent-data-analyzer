"""Planner Agent：把自然语言需求解析为结构化 TaskPlan。

输入：user_query（"分析女性用户在不同年龄层的广告点击行为"）
输出：TaskPlan（task_type / target_column / need_cleaning / relevant_tables / subtasks / rationale）
     + DatasetMeta（list_tables 拿到的表清单）

为什么用 Opus 4-7
=================
任务拆解 + 意图识别 + 输出 schema 严格的结构化结果，是 Planner 的核心，
对推理质量要求高于 worker。

为什么用 submit_task_plan tool 强制结构化输出
==============================================
- 比 "JSON-in-text + 正则解析" 健壮：schema 在 tool input_schema 里强约束
- 比让模型"思考完直接给 markdown"省 token：tool 调用只走 input_schema，
  最终 thought / 总结很短
- 简历可讲："用 Anthropic tool_use 实现结构化输出契约，避免脆性的文本 JSON 解析"
"""
from __future__ import annotations

from typing import Any

from app.agents.base import AgentRun, BaseAgent
from app.config import settings
from app.graph.state import DatasetMeta, TaskPlan


PLANNER_SYSTEM_PROMPT = """你是一个资深的数据分析任务规划师（Planner）。

**目标**：把用户的自然语言分析需求，结合数据库当前可用的表，规划成一个结构化的 TaskPlan。

**工作流程**（**严格遵守**）：
1. 调用 `list_tables` 了解所有可用表（必做，且只调用一次）
2. 可选：对你怀疑相关的表调用一次 `sample_table`（n=5）看几行样本，确认列含义
3. **最后必须调用 `submit_task_plan` 一次**，把结构化结果提交。提交后立即停止，不再调用任何工具。

**关键判断逻辑**：

| 用户意图 | task_type | need_cleaning | target_column |
|---|---|---|---|
| "看看 / 了解 / 探索 / 整体描述" | `eda_only` | `false` | `null` |
| "训练 / 预测 / 建模 / 分类 / 回归" | `modeling` | `true` | 必填（如 `clk`） |
| "对比 / 比较 / 不同 X 的 Y 表现 / X 与 Y 的差异" | `comparison` | `true` | `null` 或目标度量列名 |
| 其他模糊情况 | `general` | `true` | `null` |

**重要约束**：
- `relevant_tables` 必须来自 `list_tables` 返回的真实表名，不要编造
- `subtasks` 写 3-6 条短句，对后续 Explorer / Reporter 有指导意义
- 不要在 `submit_task_plan` 之前输出大段 Markdown 报告，那是 Reporter 的工作
- 单次任务总调用次数控制在 4 次以内（1 次 list_tables + 最多 2 次 sample_table + 1 次 submit_task_plan）

**示例**：
用户："分析女性用户在不同年龄层的广告点击率差异"
→ task_type=comparison, target_column=null, need_cleaning=true,
  relevant_tables=[user_profile, raw_sample],
  subtasks=["筛选女性用户", "按 age_level 分组", "计算各组 CTR", "对比并可视化"],
  rationale="用户要做分组对比，目标度量是 CTR（clk 比例），需 user_profile JOIN raw_sample"
"""


class PlannerAgent(BaseAgent):
    name = "planner"
    model = settings.anthropic_model_planner
    system_prompt = PLANNER_SYSTEM_PROMPT
    allowed_tools = ["list_tables", "sample_table", "submit_task_plan"]
    max_iterations = 6
    max_tokens = 2048

    # ===== 公开接口 =====
    def plan(self, user_query: str) -> tuple[TaskPlan | None, DatasetMeta | None, AgentRun]:
        """跑 Planner，返回 (TaskPlan, DatasetMeta, AgentRun)。

        - TaskPlan 从 submit_task_plan 的 tool_args 还原
        - DatasetMeta 从 list_tables 的 tool_result 还原
        - 任一缺失时返回对应 None，由上层（orchestrator）决定如何兜底
        """
        run = self.run(user_query)

        plan = self._extract_task_plan(run)
        dataset_meta = self._extract_dataset_meta(run)

        return plan, dataset_meta, run

    # ===== 私有：从 run.steps 反查关键产出 =====
    @staticmethod
    def _extract_task_plan(run: AgentRun) -> TaskPlan | None:
        """扫 run.steps，取最后一次 submit_task_plan 调用的 tool_args。"""
        last_args: dict[str, Any] | None = None
        for step in run.steps:
            if step.role == "tool_call" and step.tool_name == "submit_task_plan":
                last_args = step.tool_args
        if not last_args:
            return None
        # 用 TaskPlan TypedDict 构造（运行期 TypedDict 等价于 dict 字面量）
        return TaskPlan(
            task_type=last_args.get("task_type", "general"),
            target_column=last_args.get("target_column"),
            need_cleaning=bool(last_args.get("need_cleaning", True)),
            relevant_tables=list(last_args.get("relevant_tables", [])),
            subtasks=list(last_args.get("subtasks", [])),
            rationale=str(last_args.get("rationale", "")),
        )

    @staticmethod
    def _extract_dataset_meta(run: AgentRun) -> DatasetMeta | None:
        """从 run.steps 里抽 list_tables 的 tool_result（步骤里只存了 500 字摘要，
        所以这里需要重新调用一次 list_tables 拿全量结果）。

        注意：BaseAgent._execute_one_tool 里把 tool_result 截到 500 字，那是给
        UI 看的摘要；这里我们直接重调一次 list_tables（开销极小：只是元数据
        查询），拿完整 schema 入 state。
        """
        from app.tools.registry import execute_tool

        result = execute_tool("list_tables", {})
        if "tables" not in result:
            return None
        return DatasetMeta(tables=result["tables"])
