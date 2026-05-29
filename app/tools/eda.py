"""EDA 工具：profile_table / missing_value_summary / numeric_describe /
categorical_distribution / correlation_matrix。

EDA 默认从大表 TABLESAMPLE 抽 EDA_MAX_ROWS 行，避免百万行回内存。
对应届项目演示数据足够代表全量分布。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.tools.db import get_engine, validate_identifier

# 单次 EDA 查询最多回读这么多行（大表会自动 BERNOULLI 采样）
EDA_MAX_ROWS = 100_000


def _load_for_eda(table: str, cols: list[str] | None = None) -> pd.DataFrame:
    """加载表数据用于 EDA。大表自动 TABLESAMPLE 采样到 EDA_MAX_ROWS。"""
    validate_identifier(table)
    if cols:
        for c in cols:
            validate_identifier(c)
        col_expr = ", ".join(cols)
    else:
        col_expr = "*"

    eng = get_engine()
    with eng.connect() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
        if n <= EDA_MAX_ROWS:
            sql = f"SELECT {col_expr} FROM {table}"
        else:
            pct = (EDA_MAX_ROWS / n) * 100
            sql = (
                f"SELECT {col_expr} FROM {table} "
                f"TABLESAMPLE BERNOULLI({pct:.4f})"
            )
        df = pd.read_sql(text(sql), conn)
    return df


def profile_table(table: str) -> dict:
    """表的整体 profile：行数、列数、各列 dtype、unique 数、缺失率。"""
    df = _load_for_eda(table)
    profile = []
    for c in df.columns:
        s = df[c]
        profile.append(
            {
                "column": c,
                "dtype": str(s.dtype),
                "n_unique": int(s.nunique(dropna=False)),
                "n_missing": int(s.isna().sum()),
                "missing_pct": round(float(s.isna().mean()) * 100, 2),
            }
        )
    return {
        "table": table,
        "sampled_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": profile,
    }


def missing_value_summary(table: str) -> dict:
    """缺失值汇总：按缺失比例降序列出有缺失的列。"""
    df = _load_for_eda(table)
    rows = []
    for c in df.columns:
        n_missing = int(df[c].isna().sum())
        if n_missing > 0:
            rows.append(
                {
                    "column": c,
                    "n_missing": n_missing,
                    "missing_pct": round(n_missing / len(df) * 100, 2),
                }
            )
    rows.sort(key=lambda r: -r["missing_pct"])
    return {
        "table": table,
        "sampled_rows": int(len(df)),
        "columns_with_missing": rows,
    }


def numeric_describe(table: str, cols: list[str] | None = None) -> dict:
    """数值列描述统计：count/mean/std/min/q25/median/q75/max。"""
    df = _load_for_eda(table, cols)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return {"table": table, "error": "No numeric columns found."}
    desc = num_df.describe(percentiles=[0.25, 0.5, 0.75]).round(4)
    result = {}
    for c in num_df.columns:
        result[c] = {
            "count": int(desc.loc["count", c]),
            "mean": float(desc.loc["mean", c]),
            "std": float(desc.loc["std", c]),
            "min": float(desc.loc["min", c]),
            "q25": float(desc.loc["25%", c]),
            "median": float(desc.loc["50%", c]),
            "q75": float(desc.loc["75%", c]),
            "max": float(desc.loc["max", c]),
        }
    return {
        "table": table,
        "sampled_rows": int(len(df)),
        "numeric_stats": result,
    }


def categorical_distribution(table: str, col: str, top_k: int = 20) -> dict:
    """指定列的 top_k 取值与占比（适合类别列 / 低基数数值列）。"""
    df = _load_for_eda(table, [col])
    top_k = min(max(1, int(top_k)), 50)
    vc = df[col].value_counts(dropna=False).head(top_k)
    total = int(len(df))
    return {
        "table": table,
        "column": col,
        "sampled_rows": total,
        "n_unique": int(df[col].nunique(dropna=False)),
        "top_values": [
            {
                "value": str(k) if not pd.isna(k) else None,
                "count": int(v),
                "pct": round(v / total * 100, 2),
            }
            for k, v in vc.items()
        ],
    }


def correlation_matrix(table: str, cols: list[str]) -> dict:
    """数值列 Pearson 相关性矩阵。需至少 2 个数值列。"""
    if not cols or len(cols) < 2:
        return {"error": "Need at least 2 columns for correlation."}
    df = _load_for_eda(table, cols)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return {"error": "Need at least 2 numeric columns."}
    corr = num_df.corr().round(4)
    return {
        "table": table,
        "sampled_rows": int(len(df)),
        "columns": list(corr.columns),
        "matrix": {c: corr[c].to_dict() for c in corr.columns},
    }
