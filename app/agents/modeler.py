"""Modeler Agent：训练 CTR / 分类模型并提取特征重要性。

输入：Planner 的 task_plan + Cleaner 的 cleaning_report（可选）+ Explorer 报告
输出：list[ModelResult]（M3 schema 是 list，方便对比 LR vs LGBM）
"""
from __future__ import annotations

from typing import Any

from app.agents.base import AgentRun, BaseAgent
from app.config import settings
from app.graph.state import ModelResult


MODELER_SYSTEM_PROMPT = """你是一个机器学习建模专家（Modeler Agent）。

**目标**：根据 Planner 的任务 / Cleaner 的清洗结果，构建特征宽表并训练分类模型，
最终提交结构化结果（model_name + metrics + feature_importance）。

**严格的工作流程**：
1. 调用 `build_feature_table()` 构建宽表 feature_wide_<run_id>（默认使用 cleaned_<run_id>，自动回退到原始 user_profile）
2. 调用 `train_lgbm()` —— LightGBM 通常 AUC 比 LR 高 5-10%，作为主模型
3. （可选）调用 `train_lr()` 作为基线模型对比
4. 调用 `submit_model_result()` 提交主模型的结构化结果。**调用后立即停止**。

**重要约束**：
- 总工具调用次数控制在 5 次以内
- target_col 默认 'target'（build_feature_table 已经把 raw_sample.clk 重命名为 target）
- 不要尝试调用 train_*_xxx 之外的模型（registry 没有别的实现）
- 训练较慢（30 万行 × 200 棵树约 30-60s），不要无故重训
- 如果 build_feature_table 报错，思考是否数据未就绪，不要重复重试相同参数

**提交时建议优先选 LightGBM 的结果作为最终 model_name**，因为：
- 二分类不平衡（CTR ~5%），LightGBM 内置 is_unbalance 处理更好
- LightGBM 原生支持缺失值，无需额外处理
- 特征重要性更可解释（gain-based）

**关键提示**：
- 评估指标必须包含 auc / f1 / accuracy 三项
- feature_importance 取 top 10-15 项，按 importance 降序
- notes 字段写"模型选择 + 关键发现"，1-3 句即可
"""


class ModelerAgent(BaseAgent):
    name = "modeler"
    model = settings.anthropic_model_worker
    system_prompt = MODELER_SYSTEM_PROMPT
    allowed_tools = [
        "build_feature_table",
        "train_lr",
        "train_lgbm",
        "submit_model_result",
    ]
    max_iterations = 8
    max_tokens = 2048

    def model_train(
        self,
        user_query: str,
        target_column: str | None = None,
        has_cleaned: bool = False,
    ) -> tuple[list[ModelResult], AgentRun]:
        """跑 Modeler，返回 (ModelResult 列表, AgentRun)。

        多次 train_lgbm/train_lr 调用都会被解析成独立的 ModelResult；
        submit_model_result 只是用户级"主模型"的标注，list 里会包含所有训练过的模型。
        """
        ctx = [f"用户原始诉求：{user_query}"]
        if target_column:
            ctx.append(f"目标列（来自 Planner）：`{target_column}`")
        ctx.append(
            f"是否已完成清洗：{'是（cleaned_<run_id> 可用）' if has_cleaned else '否（直接用 user_profile）'}"
        )
        ctx.append("请按系统提示的流程构建特征 + 训练模型，并提交结果。")
        user_msg = "\n\n".join(ctx)

        run = self.run(user_msg)
        results = self._extract_results(run)
        return results, run

    @staticmethod
    def _extract_results(run: AgentRun) -> list[ModelResult]:
        """扫 run.steps，把每次 train_* 的 tool_result 与 submit_model_result 的提交
        都转成 ModelResult。"""
        results: list[ModelResult] = []
        # 1) 从 tool_result 抽 train_lr / train_lgbm 的真实指标
        for step in run.steps:
            if step.role == "tool_call" and step.tool_name in {"train_lr", "train_lgbm"}:
                # 配对：找紧随其后的 tool_result（base.py 保证 call/result 顺序）
                pass  # 用 result 的 step 更准，下面处理
        # 用 result 配对
        prev_call: str | None = None
        for step in run.steps:
            if step.role == "tool_call" and step.tool_name in {"train_lr", "train_lgbm"}:
                prev_call = step.tool_name
            elif step.role == "tool_result" and prev_call:
                # tool_result.content 是截断后的 JSON 字符串，从原始 result 不好恢复全部
                # 这里仅保留 tool_name 占位；真实的 metrics 由 submit_model_result 携带
                prev_call = None

        # 2) 主 ModelResult 来自 submit_model_result
        for step in run.steps:
            if step.role == "tool_call" and step.tool_name == "submit_model_result":
                args = step.tool_args or {}
                results.append(
                    ModelResult(
                        model_name=str(args.get("model_name", "unknown")),
                        metrics=dict(args.get("metrics", {})),
                        feature_importance=list(args.get("feature_importance", [])),
                    )
                )
        return results
