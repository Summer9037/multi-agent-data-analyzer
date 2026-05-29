"""Cleaner Agent：根据 Explorer 的发现，对单表做缺失/异常/编码处理。

输入：Explorer 的 EDA 报告 + Planner 的 task_plan + 用户原始 query
输出：CleaningReport（cleaned_table / operations / rows_before / rows_after）

为什么用 Sonnet 4-6
====================
任务模板比较固定（init → 一组 op → persist → submit），不需要 Opus 级别推理。

为什么用 submit_cleaning_report 强制结构化输出
==============================================
跟 Planner 用 submit_task_plan 一样的模式：避免文本 JSON 解析的脆性，
让 Cleaner 显式把"做了哪几步"汇总成结构化 report，便于下游 Reporter 引用。
"""
from __future__ import annotations

from typing import Any

from app.agents.base import AgentRun, BaseAgent
from app.config import settings
from app.graph.state import CleaningReport


CLEANER_SYSTEM_PROMPT = """你是一个数据清洗专家（Cleaner Agent）。

**目标**：根据 Explorer 给出的 EDA 发现，对一张目标源表做必要的清洗（缺失值 / 异常值 /
类别编码），把结果落到 cleaned_<run_id> 表，供下游 Modeler 使用。

**严格的工作流程**：
1. 调用 `init_cleaning_table(source_table=...)` —— **必须是第一步**
2. 根据 EDA 发现做 1-6 步具体清洗操作（impute_missing / cap_outliers_iqr /
   encode_categorical / drop_nulls）
3. 调用 `persist_cleaned()` 把工作表固化为 cleaned_<run_id>
4. 调用 `submit_cleaning_report(...)` 提交最终报告。**调用后立即停止**，不再调用任何工具。

**清洗策略指南**：
- 缺失率 < 5%：可以直接 drop_nulls（删除少量缺失行不影响数据规模）
- 缺失率 5%-50%：用 impute_missing（数值列用 median，类别列用 mode）
- 缺失率 > 50%：用 impute_missing(strategy='constant', value=-1) 把缺失单独标记为一个类别（"缺失本身是信号"）
- 严重右偏的数值列（mean >> median）：用 cap_outliers_iqr(col=..., k=1.5 或 3.0)
- 高基数类别列（n_unique > 100）：跳过编码，让 Modeler 决定
- 中等基数类别列（< 100）：用 encode_categorical 做 label encoding

**重要约束**：
- 总工具调用次数（不含 init/persist/submit）控制在 6 次以内
- 不要重复对同一列同一策略
- 如果工具返回 error，思考是否换列或换策略，不要重试同样的参数
- 不要在 submit_cleaning_report 之前输出大段文字，简短解释即可

**最终 submit_cleaning_report 的参数**：
- cleaned_table：persist_cleaned 返回的表名
- rows_before / rows_after：init_cleaning_table 与 persist_cleaned 各自返回的 n_rows
- operations：列出每一步操作，每项 {"op": "工具名", "detail": "一句话简述"}
"""


class CleanerAgent(BaseAgent):
    name = "cleaner"
    model = settings.anthropic_model_worker
    system_prompt = CLEANER_SYSTEM_PROMPT
    allowed_tools = [
        "init_cleaning_table",
        "drop_nulls",
        "impute_missing",
        "cap_outliers_iqr",
        "encode_categorical",
        "persist_cleaned",
        "submit_cleaning_report",
    ]
    max_iterations = 12
    max_tokens = 2048

    def clean(
        self,
        user_query: str,
        source_table: str,
        explorer_report_md: str | None = None,
    ) -> tuple[CleaningReport | None, AgentRun]:
        """跑 Cleaner，返回 (CleaningReport, AgentRun)。

        source_table 是上层（builder.cleaner_node）决定的清洗目标表。
        把 Explorer 的报告作为参考一起喂给 Cleaner。
        """
        msg_parts = [
            f"用户原始诉求：{user_query}",
            f"本次需要清洗的源表：`{source_table}`",
        ]
        if explorer_report_md:
            # Explorer 的报告可能很长，截 4000 字给 Cleaner 看缺失/分布要点
            msg_parts.append(
                "Explorer 给出的 EDA 发现（节选）：\n"
                + explorer_report_md[:4000]
            )
        msg_parts.append("请按系统提示的流程对该表做清洗，并提交报告。")
        user_msg = "\n\n".join(msg_parts)

        run = self.run(user_msg)
        return self._extract_report(run), run

    @staticmethod
    def _extract_report(run: AgentRun) -> CleaningReport | None:
        """扫 run.steps 找最后一次 submit_cleaning_report 的 tool_args。"""
        last: dict[str, Any] | None = None
        for step in run.steps:
            if step.role == "tool_call" and step.tool_name == "submit_cleaning_report":
                last = step.tool_args
        if not last:
            return None
        return CleaningReport(
            cleaned_table=str(last.get("cleaned_table", "")),
            operations=list(last.get("operations", [])),
            rows_before=int(last.get("rows_before", 0)),
            rows_after=int(last.get("rows_after", 0)),
        )
