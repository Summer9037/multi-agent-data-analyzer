"""AnalysisState：贯穿全工作流的共享状态。

设计要点
========

1. **TypedDict + reducers**：用 LangGraph 推荐的 TypedDict 写法。对于
   "多次节点追加" 的字段（charts / errors / agent_messages / model_results）
   用 `Annotated[list[...], operator.add]`，让 LangGraph 自动做列表拼接，
   而不是覆盖。这样后续 Visualizer 多次产出图表时不会互相覆盖。

2. **子结构也用 TypedDict**：TaskPlan / ExplorerOutput / CleaningReport
   等都是 TypedDict，方便 IDE 类型提示，也方便序列化到 JSON 入库。

3. **M3 范围**：Planner / Explorer / Reporter 三个节点会用到的字段。
   Cleaner / Modeler / Visualizer 相关字段保留占位，M4 再启用，避免
   重复修改 schema。
"""
from __future__ import annotations

from operator import add
from typing import Annotated, Literal, TypedDict


# ============ 子结构 ============

class TableSchema(TypedDict):
    """单表的结构信息（list_tables 返回项的精简版）。"""
    name: str
    rows: int
    columns: list[dict]  # [{"name": "...", "type": "..."}]


class TaskPlan(TypedDict):
    """Planner 输出的结构化任务计划。

    task_type:
      - "eda_only":     纯探索，跳过 Cleaner/Modeler
      - "modeling":     需要训练模型（如 CTR 预测）
      - "comparison":   分组对比分析（如男女对比）
      - "general":      默认走全流程
    """
    task_type: Literal["eda_only", "modeling", "comparison", "general"]
    target_column: str | None  # 建模任务的目标列；非建模任务为 None
    need_cleaning: bool        # 是否需要清洗步骤
    relevant_tables: list[str] # Planner 判断与本次任务相关的表
    subtasks: list[str]        # 子任务的自然语言描述（给 Explorer/Reporter 看）
    rationale: str             # Planner 的 plan 理由（用于报告"为什么这么分析"）


class DatasetMeta(TypedDict):
    """Planner 在规划前 list_tables 拿到的库内表清单（给后续节点共享）。"""
    tables: list[TableSchema]


class ExplorerOutput(TypedDict):
    """Explorer Agent 的产出。"""
    report_md: str            # 完整 EDA Markdown 报告
    tool_call_count: int      # 共调用了多少次 tool（可观测性）
    key_findings: list[str]   # M3 先留空 [], M4 让 Planner 解析后填充


class CleaningReport(TypedDict):
    """Cleaner Agent 的产出（M4）。"""
    cleaned_table: str
    operations: list[dict]
    rows_before: int
    rows_after: int


class ModelResult(TypedDict):
    """Modeler Agent 的产出（M4），可能有多个模型。"""
    model_name: str
    metrics: dict             # {"auc": 0.71, "f1": 0.18, ...}
    feature_importance: list[dict] | None  # [{"feature": "price", "imp": 0.12}, ...]


class ChartMeta(TypedDict):
    """Visualizer 产出的单张图（M4），多张图通过 reducer 累积。"""
    chart_id: str
    chart_type: Literal["bar", "hist", "heatmap", "roc", "scatter"]
    title: str
    path: str                 # outputs/charts/<run_id>/<chart_id>.html
    insight: str              # 一句 LLM 生成的洞察


class AgentMessage(TypedDict):
    """单条 Agent 执行日志，供 UI 流式渲染 + DB 归档。"""
    agent: str                # planner / explorer / ...
    role: Literal["thought", "tool_call", "tool_result", "final"]
    content: str
    tool_name: str | None
    duration_ms: int
    tokens_in: int
    tokens_out: int
    cached_tokens: int


class ErrorEntry(TypedDict):
    """节点失败的错误记录（Reporter 会在报告头部列出）。"""
    agent: str
    error_type: str
    message: str


# ============ 主 State ============

class AnalysisState(TypedDict, total=False):
    """LangGraph 工作流的共享状态。

    `total=False` 让所有字段可选，因为不同节点只关心子集，且 LangGraph
    在初始化时只需要传 user_query / selected_tables / run_id。
    """

    # --- 输入 ---
    user_query: str
    selected_tables: list[str]
    run_id: str

    # --- Planner 产出 ---
    task_plan: TaskPlan | None
    dataset_meta: DatasetMeta | None

    # --- Explorer 产出 ---
    explorer_output: ExplorerOutput | None

    # --- Cleaner 产出（M4） ---
    cleaning_report: CleaningReport | None

    # --- Modeler 产出（M4），reducer 追加 ---
    model_results: Annotated[list[ModelResult], add]

    # --- Visualizer 产出（M4），reducer 追加 ---
    charts: Annotated[list[ChartMeta], add]

    # --- Reporter 产出 ---
    report_md: str | None
    report_path: str | None   # outputs/reports/<run_id>.md

    # --- 可观测性 ---
    agent_messages: Annotated[list[AgentMessage], add]
    errors: Annotated[list[ErrorEntry], add]
    current_node: str

    # --- 累计 token 消耗（每个节点结束时自增） ---
    total_tokens_in: int
    total_tokens_out: int
    total_cached_tokens: int


def new_state(user_query: str, selected_tables: list[str], run_id: str) -> AnalysisState:
    """工厂函数：构造一个初始 state，把 reducer 字段都给定空 list。"""
    return AnalysisState(
        user_query=user_query,
        selected_tables=selected_tables,
        run_id=run_id,
        task_plan=None,
        dataset_meta=None,
        explorer_output=None,
        cleaning_report=None,
        model_results=[],
        charts=[],
        report_md=None,
        report_path=None,
        agent_messages=[],
        errors=[],
        current_node="start",
        total_tokens_in=0,
        total_tokens_out=0,
        total_cached_tokens=0,
    )
