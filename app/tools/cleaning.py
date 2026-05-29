"""数据清洗工具：

工具集（共 7 个）
==================
- init_cleaning_table(source_table)  从 source_table 复制出工作表 cleaning_work_<run_id>
- drop_nulls(cols)                   删除指定列任一为 NULL 的行
- impute_missing(col, strategy, ...) 填充缺失值（median / mean / mode / constant）
- cap_outliers_iqr(col, k)           Winsorize：把超出 [Q1-k*IQR, Q3+k*IQR] 的值截到边界
- encode_categorical(col)            label encode：用 ROW_NUMBER OVER (PARTITION BY col) 实现
- persist_cleaned()                  把工作表 RENAME 为 cleaned_<run_id>
- submit_cleaning_report(...)        提交结构化清洗报告（同 submit_task_plan 套路）

设计要点
========
- 所有工具都从 ContextVar 拿 run_id，不要求 LLM 传
- 所有操作通过 SQL 在 PG 内完成（不下载到 pandas），避免大表性能问题
- 每个工具返回 {ok: True, op: '...', detail: ...}，方便 LLM 看效果与继续推理
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import text

from app.runtime.context import get_run_id_for_sql
from app.tools.db import get_engine, validate_identifier


# === 工作表名工具 ===

def _work_table() -> str:
    return f"cleaning_work_{get_run_id_for_sql()}"


def _cleaned_table() -> str:
    return f"cleaned_{get_run_id_for_sql()}"


def _ensure_work_table_exists(conn) -> None:
    """检查工作表存在，否则抛错（除 init_cleaning_table 外的所有工具都需要）。"""
    work = _work_table()
    exists = conn.execute(
        text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name=:t)"
        ),
        {"t": work},
    ).scalar_one()
    if not exists:
        raise RuntimeError(
            f"Working table {work} not initialized. Call init_cleaning_table first."
        )


# === 工具实现 ===

def init_cleaning_table(source_table: str) -> dict:
    """从 source_table 复制出本次清洗的工作表 cleaning_work_<run_id>。

    幂等：如已存在则先 DROP。
    """
    validate_identifier(source_table)
    work = _work_table()
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {work}"))
        conn.execute(text(f"CREATE TABLE {work} AS SELECT * FROM {source_table}"))
        n = conn.execute(text(f"SELECT COUNT(*) FROM {work}")).scalar_one()
    return {
        "ok": True,
        "op": "init_cleaning_table",
        "source": source_table,
        "working_table": work,
        "n_rows": int(n),
    }


def drop_nulls(cols: list[str]) -> dict:
    """删除指定列任一为 NULL 的行。"""
    if not cols:
        return {"error": "cols cannot be empty"}
    for c in cols:
        validate_identifier(c)
    work = _work_table()
    eng = get_engine()
    where = " OR ".join(f"{c} IS NULL" for c in cols)
    with eng.begin() as conn:
        _ensure_work_table_exists(conn)
        before = conn.execute(text(f"SELECT COUNT(*) FROM {work}")).scalar_one()
        conn.execute(text(f"DELETE FROM {work} WHERE {where}"))
        after = conn.execute(text(f"SELECT COUNT(*) FROM {work}")).scalar_one()
    return {
        "ok": True,
        "op": "drop_nulls",
        "cols": cols,
        "rows_before": int(before),
        "rows_after": int(after),
        "rows_dropped": int(before - after),
    }


def impute_missing(
    col: str,
    strategy: str = "median",
    value: Any = None,
) -> dict:
    """填充指定列的缺失值。

    strategy:
      - median / mean   : 数值列
      - mode            : 用众数（任意类型）
      - constant        : 用 `value` 参数（需要 LLM 显式给出）
    """
    validate_identifier(col)
    if strategy not in {"median", "mean", "mode", "constant"}:
        return {"error": f"Unknown strategy: {strategy}"}
    if strategy == "constant" and value is None:
        return {"error": "strategy=constant requires `value`"}

    work = _work_table()
    eng = get_engine()
    with eng.begin() as conn:
        _ensure_work_table_exists(conn)

        if strategy == "median":
            fill = conn.execute(
                text(f"SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY {col}) FROM {work}")
            ).scalar()
        elif strategy == "mean":
            fill = conn.execute(text(f"SELECT AVG({col}) FROM {work}")).scalar()
        elif strategy == "mode":
            fill = conn.execute(
                text(
                    f"SELECT {col} FROM {work} WHERE {col} IS NOT NULL "
                    f"GROUP BY {col} ORDER BY COUNT(*) DESC LIMIT 1"
                )
            ).scalar()
        else:  # constant
            fill = value

        if fill is None:
            return {"error": f"Could not compute fill value for {col} with strategy={strategy}"}

        n_before = conn.execute(
            text(f"SELECT COUNT(*) FROM {work} WHERE {col} IS NULL")
        ).scalar_one()
        conn.execute(
            text(f"UPDATE {work} SET {col} = :v WHERE {col} IS NULL"),
            {"v": fill},
        )

    return {
        "ok": True,
        "op": "impute_missing",
        "col": col,
        "strategy": strategy,
        "fill_value": float(fill) if isinstance(fill, (int, float)) else str(fill),
        "n_filled": int(n_before),
    }


def cap_outliers_iqr(col: str, k: float = 1.5) -> dict:
    """Winsorize：把列 col 超出 [Q1 - k*IQR, Q3 + k*IQR] 的值截到边界。"""
    validate_identifier(col)
    if k <= 0:
        return {"error": "k must be > 0"}

    work = _work_table()
    eng = get_engine()
    with eng.begin() as conn:
        _ensure_work_table_exists(conn)
        row = conn.execute(
            text(
                f"SELECT "
                f"percentile_cont(0.25) WITHIN GROUP (ORDER BY {col}), "
                f"percentile_cont(0.75) WITHIN GROUP (ORDER BY {col}) "
                f"FROM {work}"
            )
        ).fetchone()
        if row is None or row[0] is None:
            return {"error": f"Column {col} has no numeric values"}
        q1, q3 = float(row[0]), float(row[1])
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr

        n_lower = conn.execute(
            text(f"SELECT COUNT(*) FROM {work} WHERE {col} < :lo"), {"lo": lower}
        ).scalar_one()
        n_upper = conn.execute(
            text(f"SELECT COUNT(*) FROM {work} WHERE {col} > :up"), {"up": upper}
        ).scalar_one()

        conn.execute(text(f"UPDATE {work} SET {col} = :lo WHERE {col} < :lo"), {"lo": lower})
        conn.execute(text(f"UPDATE {work} SET {col} = :up WHERE {col} > :up"), {"up": upper})

    return {
        "ok": True,
        "op": "cap_outliers_iqr",
        "col": col,
        "k": float(k),
        "q1": q1,
        "q3": q3,
        "lower_bound": lower,
        "upper_bound": upper,
        "n_capped_low": int(n_lower),
        "n_capped_high": int(n_upper),
    }


def encode_categorical(col: str) -> dict:
    """对类别列做 label encoding（用 DENSE_RANK 实现）。
    新增一列 <col>_enc，原列保留。
    """
    validate_identifier(col)
    work = _work_table()
    enc_col = f"{col}_enc"
    validate_identifier(enc_col)

    eng = get_engine()
    with eng.begin() as conn:
        _ensure_work_table_exists(conn)
        # 如果 enc 列已存在，先 drop（幂等）
        conn.execute(text(f"ALTER TABLE {work} DROP COLUMN IF EXISTS {enc_col}"))
        conn.execute(text(f"ALTER TABLE {work} ADD COLUMN {enc_col} INTEGER"))
        # 用 DENSE_RANK 计算 label encoding（NULL 映射到 0）
        # PG 不能直接 UPDATE FROM 自身，所以用临时映射子查询
        conn.execute(
            text(
                f"""
                UPDATE {work} AS t
                SET {enc_col} = m.rank
                FROM (
                    SELECT {col} AS k,
                           DENSE_RANK() OVER (ORDER BY {col} NULLS FIRST) AS rank
                    FROM (SELECT DISTINCT {col} FROM {work}) d
                ) AS m
                WHERE (t.{col} IS NOT DISTINCT FROM m.k)
                """
            )
        )
        n_unique = conn.execute(
            text(f"SELECT COUNT(DISTINCT {enc_col}) FROM {work}")
        ).scalar_one()

    return {
        "ok": True,
        "op": "encode_categorical",
        "col": col,
        "new_col": enc_col,
        "n_unique": int(n_unique),
    }


def persist_cleaned() -> dict:
    """把工作表 RENAME 为 cleaned_<run_id>，作为 Cleaner 阶段的最终产物。

    幂等：cleaned_<run_id> 已存在则先 DROP。
    """
    work = _work_table()
    cleaned = _cleaned_table()
    eng = get_engine()
    with eng.begin() as conn:
        _ensure_work_table_exists(conn)
        conn.execute(text(f"DROP TABLE IF EXISTS {cleaned}"))
        conn.execute(text(f"ALTER TABLE {work} RENAME TO {cleaned}"))
        n = conn.execute(text(f"SELECT COUNT(*) FROM {cleaned}")).scalar_one()
    return {
        "ok": True,
        "op": "persist_cleaned",
        "cleaned_table": cleaned,
        "n_rows": int(n),
    }


def submit_cleaning_report(
    cleaned_table: str,
    rows_before: int,
    rows_after: int,
    operations: list[dict],
) -> dict:
    """Cleaner 调用本工具提交最终的清洗报告（同 submit_task_plan 套路）。"""
    return {
        "ok": True,
        "received": {
            "cleaned_table": cleaned_table,
            "rows_before": int(rows_before),
            "rows_after": int(rows_after),
            "operations": list(operations or []),
        },
    }
