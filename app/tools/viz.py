"""可视化工具：从 PG 表生成 Plotly 图表（HTML 文件 + 一句洞察）。

工具集（5 个）
==============
- plotly_hist(table, col)              单列直方图
- plotly_bar_topk(table, col, top_k)   类别列 Top K 占比柱状图
- plotly_grouped_ctr(table, group_col, target_col)  按 group 计算 CTR 的对比柱状图
- plotly_corr_heatmap(table, cols)     数值列相关性热力图
- submit_chart(...)                    Visualizer 提交一张图的元信息（含一句 LLM 洞察）

输出位置
========
所有 HTML 文件落到 outputs/charts/<run_id>/<chart_id>.html
chart_id 由 Visualizer 自定义，常见为 'gender_ctr' / 'price_hist' 等
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sqlalchemy import text

from app.config import settings
from app.runtime.context import get_run_id
from app.tools.db import get_engine, validate_identifier


_DEFAULT_HEIGHT = 400
# 商务浅蓝主色调（Office / Tableau 类 BI 风格）
_BIZ_BLUE = "#5B9BD5"
_BIZ_TEMPLATE = "plotly_white"


def _apply_biz_style(fig, height: int | None = None) -> None:
    """统一商务样式：白底 + 浅蓝主色 + 紧凑边距。"""
    fig.update_layout(
        template=_BIZ_TEMPLATE,
        height=height or _DEFAULT_HEIGHT,
        margin=dict(l=40, r=20, t=50, b=40),
        font=dict(family="Segoe UI, Helvetica, Arial", size=12, color="#333"),
        title_font=dict(size=14, color="#333"),
    )


# === 路径工具 ===

def _chart_dir() -> Path:
    d = Path(settings.output_dir) / "charts" / get_run_id()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_fig(fig, chart_id: str) -> str:
    """保存 plotly figure 为 standalone HTML，返回绝对路径字符串。"""
    validate_identifier(chart_id)
    path = _chart_dir() / f"{chart_id}.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    return str(path.resolve())


# === 工具实现 ===

def plotly_hist(
    table: str,
    col: str,
    chart_id: str,
    bins: int = 40,
    title: str | None = None,
    log_x: bool = False,
) -> dict:
    """单列直方图。log_x=True 时 x 轴对数刻度（适合右偏分布如 price）。"""
    validate_identifier(table)
    validate_identifier(col)
    bins = max(5, min(int(bins), 200))

    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(
            text(f"SELECT {col} FROM {table} WHERE {col} IS NOT NULL"), conn
        )
    if df.empty:
        return {"error": f"No non-null values in {table}.{col}"}

    title = title or f"Distribution of {col}"
    fig = px.histogram(
        df, x=col, nbins=bins, title=title, log_x=log_x,
        color_discrete_sequence=[_BIZ_BLUE],
    )
    _apply_biz_style(fig)
    path = _save_fig(fig, chart_id)

    return {
        "ok": True,
        "op": "plotly_hist",
        "chart_id": chart_id,
        "path": path,
        "n_values": int(len(df)),
    }


def plotly_bar_topk(
    table: str,
    col: str,
    chart_id: str,
    top_k: int = 15,
    title: str | None = None,
) -> dict:
    """类别列 Top K 取值的频次柱状图。"""
    validate_identifier(table)
    validate_identifier(col)
    top_k = max(2, min(int(top_k), 50))

    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(
            text(
                f"SELECT {col}::text AS key, COUNT(*) AS cnt "
                f"FROM {table} GROUP BY {col} ORDER BY cnt DESC LIMIT :k"
            ),
            conn,
            params={"k": top_k},
        )
    if df.empty:
        return {"error": f"No data in {table}.{col}"}

    title = title or f"Top {top_k} values of {col}"
    fig = px.bar(
        df, x="key", y="cnt", title=title, labels={"key": col, "cnt": "count"},
        color_discrete_sequence=[_BIZ_BLUE],
    )
    _apply_biz_style(fig)
    path = _save_fig(fig, chart_id)
    return {
        "ok": True,
        "op": "plotly_bar_topk",
        "chart_id": chart_id,
        "path": path,
        "n_categories": int(len(df)),
    }


def plotly_grouped_ctr(
    table: str,
    group_col: str,
    target_col: str,
    chart_id: str,
    title: str | None = None,
) -> dict:
    """按 group_col 计算 target_col 的均值（CTR / 转化率类指标）的柱状图。

    例：plotly_grouped_ctr('raw_sample', 'pid', 'clk', 'ctr_by_pid')
        → 计算每个 pid 上的 clk 均值（即各广告位的 CTR）
    """
    validate_identifier(table)
    validate_identifier(group_col)
    validate_identifier(target_col)

    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(
            text(
                f"SELECT {group_col}::text AS key, "
                f"AVG({target_col}::float) AS ctr, COUNT(*) AS n "
                f"FROM {table} "
                f"WHERE {group_col} IS NOT NULL "
                f"GROUP BY {group_col} ORDER BY ctr DESC"
            ),
            conn,
        )
    if df.empty:
        return {"error": f"No data for {table}.{group_col}"}

    title = title or f"{target_col} rate by {group_col}"
    fig = px.bar(
        df,
        x="key",
        y="ctr",
        hover_data=["n"],
        title=title,
        labels={"key": group_col, "ctr": f"{target_col} rate"},
        color_discrete_sequence=[_BIZ_BLUE],
    )
    _apply_biz_style(fig)
    fig.update_layout(yaxis_tickformat=".2%")
    path = _save_fig(fig, chart_id)
    return {
        "ok": True,
        "op": "plotly_grouped_ctr",
        "chart_id": chart_id,
        "path": path,
        "n_groups": int(len(df)),
        "top_3_groups": df.head(3).to_dict(orient="records"),
    }


def plotly_corr_heatmap(
    table: str,
    cols: list[str],
    chart_id: str,
    title: str | None = None,
) -> dict:
    """数值列 Pearson 相关性热力图。至少 2 列。"""
    if not cols or len(cols) < 2:
        return {"error": "Need at least 2 columns"}
    validate_identifier(table)
    for c in cols:
        validate_identifier(c)
    col_expr = ", ".join(cols)

    eng = get_engine()
    with eng.connect() as conn:
        # 大表 BERNOULLI 抽样到 100k 行内
        n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
        if n > 100_000:
            pct = 100_000 / n * 100
            sql = f"SELECT {col_expr} FROM {table} TABLESAMPLE BERNOULLI({pct:.4f})"
        else:
            sql = f"SELECT {col_expr} FROM {table}"
        df = pd.read_sql(text(sql), conn)

    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return {"error": "Less than 2 numeric columns after loading"}

    corr = num.corr().round(3)
    title = title or "Correlation heatmap"
    # 相关性是 [-1, 1] 双向，仍用蓝-白-红发散色板，但偏向蓝色
    # （正相关用浅蓝→深蓝，负相关用红，符合主色调）
    fig = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, title=title,
    )
    _apply_biz_style(fig, height=max(_DEFAULT_HEIGHT, 60 + 30 * len(corr)))
    path = _save_fig(fig, chart_id)
    return {
        "ok": True,
        "op": "plotly_corr_heatmap",
        "chart_id": chart_id,
        "path": path,
        "columns": list(corr.columns),
    }


def submit_chart(
    chart_id: str,
    chart_type: str,
    title: str,
    path: str,
    insight: str,
) -> dict:
    """Visualizer 把一张图的元信息（含一句洞察）提交给状态。

    每张图调用一次 submit_chart；调用顺序决定 reducer 列表的顺序。
    """
    return {
        "ok": True,
        "received": {
            "chart_id": str(chart_id),
            "chart_type": str(chart_type),
            "title": str(title),
            "path": str(path),
            "insight": str(insight),
        },
    }
