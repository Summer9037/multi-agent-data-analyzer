"""Visualizer Agent：基于 Explorer / Cleaner / Modeler 的产物，生成 3-5 张图表，
每张附一句 LLM 洞察。

输出：list[ChartMeta]（M3 state schema 是 reducer 列表，每次 submit_chart 追加一条）
"""
from __future__ import annotations

from typing import Any

from app.agents.base import AgentRun, BaseAgent
from app.config import settings
from app.graph.state import ChartMeta


VISUALIZER_SYSTEM_PROMPT = """你是一个数据可视化专家（Visualizer Agent）。

**目标**：根据用户的分析需求与上游 Agent（Explorer/Cleaner/Modeler）的发现，
生成 3-5 张能直观体现核心结论的 Plotly 图表，每张图附一句具体的业务洞察。

**严格的工作流程**：
1. 选择你要画的图类型（plotly_hist / plotly_bar_topk / plotly_grouped_ctr / plotly_corr_heatmap）
2. 调用对应的 plotly_* 工具生成图，得到 path 与 chart_id
3. **紧跟着**调用 submit_chart 把这张图的 chart_id/chart_type/title/path/insight 提交到 state
4. 重复 2-3 步直到画完 3-5 张图，然后停止（不再调用工具，可输出一句简短总结）

**图表选择策略**：
- 对**右偏数值列**（如 price）→ plotly_hist + log_x=true
- 对**类别列分布**（gender / age_level / pid）→ plotly_bar_topk
- 对**分组 CTR/转化率比较**（按 gender / age_level / pid 看 clk 比例）→ plotly_grouped_ctr
- 对**多个数值列相关性** → plotly_corr_heatmap

**Insight 字段（每张图都必须写）的要求**：
- **具体**：必须含数字 / 排名 / 倍数（如"女性 CTR 5.8% 高于男性 4.2%，相对高出 38%"）
- **避免空话**："数据分布广泛 / 用户偏好多样 / 需要进一步分析"  这种废话不要
- 10-50 字一句话，直接陈述发现

**重要约束**：
- 单次任务总工具调用不超过 12 次（最多 6 张图 × 2 个工具）
- 同一图表类型不要重复画太多（同一 col 画两次直方图就重复了）
- chart_id 必须是合法的标识符（字母数字下划线，如 'price_dist'、'ctr_by_gender'），用作文件名
- 完成画图后不要再调用任何工具，直接结束

**优先画的图**（如果是 CTR 任务）：
1. 整体 CTR 分布 / 各广告位 CTR 对比（plotly_grouped_ctr on raw_sample/pid/clk）
2. 按性别的 CTR 对比 + 按年龄段的 CTR 对比（需先确认表，可能是 feature_wide_<run_id>）
3. 价格的分布（log_x）+ 主要类别列的 Top K
4. 关键数值特征的相关性热力图
"""


class VisualizerAgent(BaseAgent):
    name = "visualizer"
    model = settings.anthropic_model_worker
    system_prompt = VISUALIZER_SYSTEM_PROMPT
    allowed_tools = [
        "plotly_hist",
        "plotly_bar_topk",
        "plotly_grouped_ctr",
        "plotly_corr_heatmap",
        "submit_chart",
    ]
    max_iterations = 14
    max_tokens = 2048

    def visualize(
        self,
        user_query: str,
        task_plan: dict | None = None,
        feature_table: str | None = None,
    ) -> tuple[list[ChartMeta], AgentRun]:
        """跑 Visualizer，返回 (ChartMeta 列表, AgentRun)。"""
        ctx = [f"用户原始诉求：{user_query}"]
        if task_plan:
            ctx.append(
                "Planner 给出的相关表："
                + ", ".join(task_plan.get("relevant_tables", []) or [])
            )
        if feature_table:
            ctx.append(
                f"Modeler 已构建宽表 `{feature_table}`，里面 `target` 列就是 raw_sample.clk，"
                f"可以直接对它做 plotly_grouped_ctr(target_col='target')。"
            )
        ctx.append("可用的数据源还有原始表：raw_sample / ad_feature / user_profile")
        ctx.append("请按系统提示选 3-5 张最有价值的图，每张图后立刻 submit_chart。")
        user_msg = "\n\n".join(ctx)

        run = self.run(user_msg)
        charts = self._extract_charts(run)
        return charts, run

    @staticmethod
    def _extract_charts(run: AgentRun) -> list[ChartMeta]:
        out: list[ChartMeta] = []
        for step in run.steps:
            if step.role == "tool_call" and step.tool_name == "submit_chart":
                args = step.tool_args or {}
                out.append(
                    ChartMeta(
                        chart_id=str(args.get("chart_id", "")),
                        chart_type=str(args.get("chart_type", "bar")),  # type: ignore[arg-type]
                        title=str(args.get("title", "")),
                        path=str(args.get("path", "")),
                        insight=str(args.get("insight", "")),
                    )
                )
        return out
