"""数据库工具：list_tables / sample_table / query_sql。

所有函数返回纯 dict / list[dict]，方便序列化给 LLM。
绝不把原始 DataFrame 直接返回——pandas 对象不能 JSON 序列化，且大表会爆 token。
"""
from __future__ import annotations

import re

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from app.config import settings

# 业务表白名单（避免 LLM 试图查 runtime 表导致循环引用）
BUSINESS_TABLES: list[str] = ["ad_feature", "raw_sample", "user_profile"]

# SQL 关键字黑名单（query_sql 用）
_FORBIDDEN_SQL_KEYWORDS = (
    "DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE",
    "ALTER", "CREATE", "GRANT", "REVOKE", "VACUUM",
)

_engine: Engine | None = None


def get_engine() -> Engine:
    """进程内单例 SQLAlchemy engine。"""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.db_url, connect_args=settings.db_connect_args
        )
    return _engine


def validate_identifier(name: str) -> None:
    """校验标识符（表名/列名）只含字母数字下划线，防 SQL 注入。"""
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
        raise ValueError(f"Invalid identifier: {name!r}")


def list_tables() -> dict:
    """列出业务表的名称、行数与列定义。

    返回 {tables: [{name, rows, columns: [{name, type}]}]}
    """
    eng = get_engine()
    inspector = inspect(eng)
    out = []
    with eng.connect() as conn:
        for name in BUSINESS_TABLES:
            if not inspector.has_table(name):
                continue
            cols = [
                {"name": c["name"], "type": str(c["type"])}
                for c in inspector.get_columns(name)
            ]
            rows = conn.execute(text(f"SELECT COUNT(*) FROM {name}")).scalar_one()
            out.append({"name": name, "rows": int(rows), "columns": cols})
    return {"tables": out}


def sample_table(table: str, n: int = 5) -> dict:
    """从指定表随机抽样 n 行作为预览。n 最大 50。"""
    validate_identifier(table)
    n = min(max(1, int(n)), 50)
    eng = get_engine()
    with eng.connect() as conn:
        df = pd.read_sql(
            text(f"SELECT * FROM {table} ORDER BY random() LIMIT :n"),
            conn,
            params={"n": n},
        )
    return {
        "table": table,
        "n_rows": len(df),
        "records": df.to_dict(orient="records"),
    }


def query_sql(sql: str, max_rows: int = 100) -> dict:
    """执行只读 SQL（只允许 SELECT / WITH 开头），有黑名单关键字保护。

    max_rows 限制返回行数（最大 10000）。超过会截断并返回 truncated=True。
    """
    sql_clean = sql.strip().rstrip(";")
    upper = sql_clean.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return {"error": "Only SELECT or WITH queries are allowed."}
    for kw in _FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            return {"error": f"Query contains forbidden keyword: {kw}"}

    max_rows = min(max(1, int(max_rows)), 10_000)
    eng = get_engine()
    try:
        with eng.connect() as conn:
            df = pd.read_sql(text(sql_clean), conn)
    except Exception as e:
        return {"error": f"Query failed: {type(e).__name__}: {e}"}

    truncated = len(df) > max_rows
    if truncated:
        df = df.head(max_rows)
    return {
        "sql": sql_clean,
        "n_rows": int(len(df)),
        "truncated": bool(truncated),
        "columns": list(df.columns),
        "records": df.to_dict(orient="records"),
    }
