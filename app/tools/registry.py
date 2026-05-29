"""工具注册表：TOOL_SPECS (Anthropic JSON Schema) + execute_tool 分发器。

新增工具的流程：
1. 在 db.py / eda.py / ... 写纯 Python 函数
2. 在 TOOL_SPECS 加上 Anthropic 兼容的 JSON Schema 描述
3. 在 TOOL_IMPL 添加 name → 函数的映射
"""
from __future__ import annotations

from typing import Any, Callable

from app.tools import cleaning as cleaning_tools
from app.tools import db as db_tools
from app.tools import eda as eda_tools
from app.tools import modeling as modeling_tools
from app.tools import viz as viz_tools


TOOL_SPECS: list[dict] = [
    # === DB tools ===
    {
        "name": "list_tables",
        "description": (
            "列出数据库中所有业务表的名称、行数、列定义。"
            "应当作为任何探索任务的第一步。"
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "sample_table",
        "description": (
            "从指定表随机抽样 n 行作为预览，用于直观了解每列长什么样。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "表名"},
                "n": {
                    "type": "integer",
                    "description": "采样行数，默认 5，最大 50",
                    "default": 5,
                },
            },
            "required": ["table"],
        },
    },
    {
        "name": "query_sql",
        "description": (
            "执行只读 SQL。仅允许 SELECT/WITH，禁止 DROP/DELETE/UPDATE 等。"
            "返回行数受 max_rows 限制（默认 100，最大 10000）。"
            "用于自定义的 JOIN / GROUP BY / 条件过滤等灵活查询。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SELECT 或 WITH 开头的 SQL"},
                "max_rows": {
                    "type": "integer",
                    "description": "最多返回多少行，默认 100",
                    "default": 100,
                },
            },
            "required": ["sql"],
        },
    },
    # === EDA tools ===
    {
        "name": "profile_table",
        "description": (
            "表的整体 profile：行数、列数、每列的 dtype、unique 数、缺失率。"
            "用于初步了解表结构。大表会自动采样到 10 万行。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {"table": {"type": "string", "description": "表名"}},
            "required": ["table"],
        },
    },
    {
        "name": "missing_value_summary",
        "description": "统计指定表每列的缺失值数量与占比（按比例降序）。",
        "input_schema": {
            "type": "object",
            "properties": {"table": {"type": "string", "description": "表名"}},
            "required": ["table"],
        },
    },
    {
        "name": "numeric_describe",
        "description": (
            "数值列描述统计：count/mean/std/min/q25/median/q75/max。"
            "不传 cols 则对全部数值列计算。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "表名"},
                "cols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "可选，指定要分析的列；不传则全部数值列",
                },
            },
            "required": ["table"],
        },
    },
    {
        "name": "categorical_distribution",
        "description": (
            "某列 top_k 取值与占比，适合类别列或低基数数值列（性别、年龄段等）。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "表名"},
                "col": {"type": "string", "description": "要分析的列名"},
                "top_k": {
                    "type": "integer",
                    "description": "返回前 K 个值，默认 20，最大 50",
                    "default": 20,
                },
            },
            "required": ["table", "col"],
        },
    },
    {
        "name": "correlation_matrix",
        "description": (
            "数值列之间的 Pearson 相关性矩阵。需至少 2 个数值列。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "表名"},
                "cols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要计算相关性的列列表（至少 2 个数值列）",
                },
            },
            "required": ["table", "cols"],
        },
    },
    # === Submission tools（结构化输出契约）===
    # Planner 强制通过 submit_task_plan 输出 TaskPlan，避免文本 JSON 解析的脆性。
    # 工具实现就是个回声器，PlannerAgent 在循环结束后从 run.steps 取最后一次
    # submit_task_plan 的 tool_args 作为最终 plan。
    {
        "name": "submit_task_plan",
        "description": (
            "提交最终的任务规划。调用本工具即视为规划完成，调用一次后请直接结束，"
            "不要再调用其他工具。所有字段都必须填写。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "enum": ["eda_only", "modeling", "comparison", "general"],
                    "description": (
                        "任务类型："
                        "eda_only=纯探索；"
                        "modeling=需训练预测模型（如 CTR）；"
                        "comparison=分组对比分析；"
                        "general=默认全流程"
                    ),
                },
                "target_column": {
                    "type": ["string", "null"],
                    "description": "建模任务的目标列名（如 clk）；非建模任务填 null",
                },
                "need_cleaning": {
                    "type": "boolean",
                    "description": "是否需要走 Cleaner 节点做缺失值/异常值处理",
                },
                "relevant_tables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "本任务相关的表名列表（必须是 list_tables 返回的表）",
                },
                "subtasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "拆解的子任务自然语言描述，3-6 条",
                },
                "rationale": {
                    "type": "string",
                    "description": "选择这个 task_type / 子任务的理由（1-3 句）",
                },
            },
            "required": [
                "task_type",
                "target_column",
                "need_cleaning",
                "relevant_tables",
                "subtasks",
                "rationale",
            ],
        },
    },
]


def _submit_task_plan(**kwargs: Any) -> dict:
    """submit_task_plan 的实现：原样回声 + 状态确认。

    真正"读取" Planner 提交的 plan 的逻辑在 PlannerAgent，它会扫
    run.steps 找最后一次 tool_call(name='submit_task_plan') 的 tool_args。
    """
    return {"status": "accepted", "received": kwargs}


# === 把 cleaning tools 的 spec 追加到 TOOL_SPECS ===
TOOL_SPECS += [
    {
        "name": "init_cleaning_table",
        "description": (
            "把 source_table 完整复制到本次运行的工作表 cleaning_work_<run_id>，"
            "后续所有清洗操作都作用在工作表上。必须作为清洗流程的第一步。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source_table": {"type": "string", "description": "要清洗的源表名"},
            },
            "required": ["source_table"],
        },
    },
    {
        "name": "drop_nulls",
        "description": "删除工作表中指定列任一为 NULL 的行。",
        "input_schema": {
            "type": "object",
            "properties": {
                "cols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要检查 NULL 的列名列表",
                },
            },
            "required": ["cols"],
        },
    },
    {
        "name": "impute_missing",
        "description": (
            "用指定策略填充工作表某列的缺失值。"
            "median/mean 仅适合数值列；mode 适合任意类型；"
            "constant 需要显式传 value。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "col": {"type": "string", "description": "目标列名"},
                "strategy": {
                    "type": "string",
                    "enum": ["median", "mean", "mode", "constant"],
                    "description": "填充策略",
                },
                "value": {
                    "description": "当 strategy=constant 时使用的填充值",
                },
            },
            "required": ["col", "strategy"],
        },
    },
    {
        "name": "cap_outliers_iqr",
        "description": (
            "Winsorize：把工作表 col 列超出 [Q1 - k*IQR, Q3 + k*IQR] 的值"
            "截断到边界，而不是直接删除行。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "col": {"type": "string", "description": "目标数值列"},
                "k": {
                    "type": "number",
                    "description": "IQR 倍数，默认 1.5（更激进可用 3.0）",
                    "default": 1.5,
                },
            },
            "required": ["col"],
        },
    },
    {
        "name": "encode_categorical",
        "description": (
            "对类别列做 label encoding（DENSE_RANK），新增 <col>_enc 列。"
            "适合中等基数（< 100）的类别列；高基数（如 user_id）不适合。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "col": {"type": "string", "description": "要编码的类别列"},
            },
            "required": ["col"],
        },
    },
    {
        "name": "persist_cleaned",
        "description": (
            "把工作表 RENAME 为 cleaned_<run_id>，作为本次清洗的最终产物，"
            "下游 Modeler 会从这张表读数据。必须作为清洗流程的最后一步。"
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    # === Modeling tools ===
    {
        "name": "build_feature_table",
        "description": (
            "把 raw_sample / ad_feature / user_profile（或 cleaned_<run_id>）通过 "
            "adgroup_id 与 user_id JOIN 成宽表 feature_wide_<run_id>，作为后续训练的输入。"
            "默认 use_cleaned_user_table=True：如果存在 cleaned_<run_id> 则优先使用。"
            "目标列在新表里统一叫 `target`。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_col": {
                    "type": "string",
                    "description": "raw_sample 里要预测的二分类列，默认 'clk'",
                    "default": "clk",
                },
                "use_cleaned_user_table": {
                    "type": "boolean",
                    "description": "是否优先用 cleaned_<run_id> 替代 user_profile",
                    "default": True,
                },
            },
        },
    },
    {
        "name": "train_lr",
        "description": (
            "在 feature_wide_<run_id> 上训练 LogisticRegression，"
            "stratified 80/20 切分，返回 metrics 与 top-15 重要特征。"
            "作为基线模型存在，速度快但表达力有限。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_col": {"type": "string", "default": "target"},
                "test_size": {"type": "number", "default": 0.2},
            },
        },
    },
    {
        "name": "train_lgbm",
        "description": (
            "在 feature_wide_<run_id> 上训练 LightGBM（GBDT）二分类，"
            "stratified 80/20 切分 + is_unbalance=True 处理类别不平衡，"
            "返回 metrics 与 top-15 重要特征。通常 AUC 比 LR 高 5-10%。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_col": {"type": "string", "default": "target"},
                "test_size": {"type": "number", "default": 0.2},
                "n_estimators": {"type": "integer", "default": 200},
            },
        },
    },
    {
        "name": "submit_model_result",
        "description": (
            "Modeler 提交结构化的最终模型结果（model_name / metrics / "
            "feature_importance / notes）。调用后立即停止，不再调任何工具。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "模型名（如 'lightgbm' / 'logistic_regression'）",
                },
                "metrics": {
                    "type": "object",
                    "description": "评估指标 dict，必须含 auc / f1 / accuracy 三项",
                },
                "feature_importance": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "feature": {"type": "string"},
                            "importance": {"type": "number"},
                        },
                        "required": ["feature", "importance"],
                    },
                    "description": "Top 10-15 重要特征列表",
                },
                "notes": {
                    "type": "string",
                    "description": "模型选择 / 调参的简短说明（1-3 句）",
                },
            },
            "required": ["model_name", "metrics", "feature_importance"],
        },
    },
    # === Visualization tools ===
    {
        "name": "plotly_hist",
        "description": (
            "对指定表的一个数值列画直方图。右偏分布（如 price）请设 log_x=true。"
            "chart_id 用作 HTML 文件名（如 'price_hist'）。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "col": {"type": "string"},
                "chart_id": {"type": "string", "description": "字母数字下划线，作文件名"},
                "bins": {"type": "integer", "default": 40},
                "title": {"type": "string"},
                "log_x": {"type": "boolean", "default": False},
            },
            "required": ["table", "col", "chart_id"],
        },
    },
    {
        "name": "plotly_bar_topk",
        "description": "类别列 Top K 频次柱状图。适合 pid / cate_id / age_level 等。",
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "col": {"type": "string"},
                "chart_id": {"type": "string"},
                "top_k": {"type": "integer", "default": 15},
                "title": {"type": "string"},
            },
            "required": ["table", "col", "chart_id"],
        },
    },
    {
        "name": "plotly_grouped_ctr",
        "description": (
            "按 group_col 分组计算 target_col 的均值（即 CTR / 转化率）的柱状图。"
            "如 plotly_grouped_ctr(raw_sample, pid, clk) 看不同广告位 CTR 差异。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "group_col": {"type": "string"},
                "target_col": {"type": "string"},
                "chart_id": {"type": "string"},
                "title": {"type": "string"},
            },
            "required": ["table", "group_col", "target_col", "chart_id"],
        },
    },
    {
        "name": "plotly_corr_heatmap",
        "description": "对指定表的多列数值列画 Pearson 相关性热力图。",
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "cols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "至少 2 个数值列",
                },
                "chart_id": {"type": "string"},
                "title": {"type": "string"},
            },
            "required": ["table", "cols", "chart_id"],
        },
    },
    {
        "name": "submit_chart",
        "description": (
            "把一张已经生成的图的元信息（chart_id / type / title / path / insight）"
            "提交进 state.charts。**每张图都要紧跟着调用一次 submit_chart**，"
            "insight 字段写一句话总结该图的业务发现（避免空话）。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chart_id": {"type": "string"},
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "hist", "heatmap", "roc", "scatter"],
                },
                "title": {"type": "string"},
                "path": {"type": "string", "description": "plotly_* 工具返回的 path"},
                "insight": {
                    "type": "string",
                    "description": "一句话洞察，针对图中可见的业务现象（10-50 字）",
                },
            },
            "required": ["chart_id", "chart_type", "title", "path", "insight"],
        },
    },
    {
        "name": "submit_cleaning_report",
        "description": (
            "提交结构化的清洗报告。调用后即视为清洗阶段完成，不要再调用其他工具。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cleaned_table": {
                    "type": "string",
                    "description": "persist_cleaned 产出的表名（如 cleaned_xxx）",
                },
                "rows_before": {
                    "type": "integer",
                    "description": "清洗前的行数（init_cleaning_table 返回的 n_rows）",
                },
                "rows_after": {
                    "type": "integer",
                    "description": "清洗后的行数（persist_cleaned 返回的 n_rows）",
                },
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string"},
                            "detail": {"type": "string"},
                        },
                        "required": ["op", "detail"],
                    },
                    "description": "本次清洗执行过的操作摘要，每项含 op 名 + 一句简述",
                },
            },
            "required": ["cleaned_table", "rows_before", "rows_after", "operations"],
        },
    },
]


TOOL_IMPL: dict[str, Callable[..., dict]] = {
    "list_tables": db_tools.list_tables,
    "sample_table": db_tools.sample_table,
    "query_sql": db_tools.query_sql,
    "profile_table": eda_tools.profile_table,
    "missing_value_summary": eda_tools.missing_value_summary,
    "numeric_describe": eda_tools.numeric_describe,
    "categorical_distribution": eda_tools.categorical_distribution,
    "correlation_matrix": eda_tools.correlation_matrix,
    "submit_task_plan": _submit_task_plan,
    # cleaning tools
    "init_cleaning_table": cleaning_tools.init_cleaning_table,
    "drop_nulls": cleaning_tools.drop_nulls,
    "impute_missing": cleaning_tools.impute_missing,
    "cap_outliers_iqr": cleaning_tools.cap_outliers_iqr,
    "encode_categorical": cleaning_tools.encode_categorical,
    "persist_cleaned": cleaning_tools.persist_cleaned,
    "submit_cleaning_report": cleaning_tools.submit_cleaning_report,
    # modeling tools
    "build_feature_table": modeling_tools.build_feature_table,
    "train_lr": modeling_tools.train_lr,
    "train_lgbm": modeling_tools.train_lgbm,
    "submit_model_result": modeling_tools.submit_model_result,
    # viz tools
    "plotly_hist": viz_tools.plotly_hist,
    "plotly_bar_topk": viz_tools.plotly_bar_topk,
    "plotly_grouped_ctr": viz_tools.plotly_grouped_ctr,
    "plotly_corr_heatmap": viz_tools.plotly_corr_heatmap,
    "submit_chart": viz_tools.submit_chart,
}


def execute_tool(name: str, args: dict[str, Any]) -> dict:
    """执行工具，捕获异常返回 dict 形式的错误。"""
    if name not in TOOL_IMPL:
        return {"error": f"Unknown tool: {name}"}
    try:
        return TOOL_IMPL[name](**(args or {}))
    except TypeError as e:
        return {"error": f"Bad arguments for {name}: {e}"}
    except Exception as e:
        return {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}
