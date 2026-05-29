"""初始化 PostgreSQL：

1. 确保目标数据库 (settings.db_name) 存在，不存在则创建。
2. 在目标数据库上执行 DDL，建表 + 索引（幂等，可重复执行）。

用法：
    python scripts/init_db.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 让脚本能 import app.*
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg
from psycopg import sql
from sqlalchemy import create_engine, text

from app.config import settings


# === 表 DDL ===
# 业务表 3 张：ad_feature / raw_sample / user_profile
# Runtime 表 2 张：analysis_runs / agent_execution_logs
# 所有列除主键外允许 NULL（原始数据有缺失）。
DDL_STATEMENTS: list[str] = [
    # --- 业务表 ---
    """
    CREATE TABLE IF NOT EXISTS ad_feature (
        adgroup_id   BIGINT PRIMARY KEY,
        cate_id      INTEGER,
        campaign_id  BIGINT,
        customer     BIGINT,
        brand        BIGINT,
        price        NUMERIC(12, 2)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ad_feature_cate  ON ad_feature(cate_id)",
    "CREATE INDEX IF NOT EXISTS idx_ad_feature_brand ON ad_feature(brand)",

    """
    CREATE TABLE IF NOT EXISTS raw_sample (
        id          BIGSERIAL PRIMARY KEY,
        user_id     BIGINT      NOT NULL,
        time_stamp  BIGINT      NOT NULL,
        adgroup_id  BIGINT      NOT NULL,
        pid         VARCHAR(32),
        nonclk      SMALLINT,
        clk         SMALLINT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_raw_sample_user ON raw_sample(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_raw_sample_ad   ON raw_sample(adgroup_id)",
    "CREATE INDEX IF NOT EXISTS idx_raw_sample_clk  ON raw_sample(clk)",

    """
    CREATE TABLE IF NOT EXISTS user_profile (
        userid                BIGINT PRIMARY KEY,
        cms_segid             INTEGER,
        cms_group_id          INTEGER,
        final_gender_code     SMALLINT,
        age_level             SMALLINT,
        pvalue_level          SMALLINT,
        shopping_level        SMALLINT,
        occupation            SMALLINT,
        new_user_class_level  SMALLINT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_user_profile_gender ON user_profile(final_gender_code)",
    "CREATE INDEX IF NOT EXISTS idx_user_profile_age    ON user_profile(age_level)",

    # --- Runtime 表 ---
    """
    CREATE TABLE IF NOT EXISTS analysis_runs (
        run_id            UUID PRIMARY KEY,
        user_query        TEXT NOT NULL,
        task_type         VARCHAR(32),
        status            VARCHAR(16) NOT NULL,
        selected_tables   JSONB,
        started_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        finished_at       TIMESTAMPTZ,
        total_tokens      INTEGER DEFAULT 0,
        cached_tokens     INTEGER DEFAULT 0,
        final_report_path TEXT,
        error_message     TEXT
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS agent_execution_logs (
        log_id              BIGSERIAL PRIMARY KEY,
        run_id              UUID        NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
        agent_name          VARCHAR(32) NOT NULL,
        step_index          INTEGER     NOT NULL,
        role                VARCHAR(16) NOT NULL,
        tool_name           VARCHAR(64),
        tool_args           JSONB,
        tool_result_summary TEXT,
        duration_ms         INTEGER,
        tokens_in           INTEGER,
        tokens_out          INTEGER,
        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_agent_logs_run ON agent_execution_logs(run_id, step_index)",
]


def ensure_database() -> None:
    """连 postgres 系统库，如果目标库不存在则创建。"""
    conn = psycopg.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        dbname="postgres",
        autocommit=True,
        **settings.db_connect_args,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (settings.db_name,),
            )
            if cur.fetchone() is None:
                cur.execute(
                    sql.SQL("CREATE DATABASE {}").format(
                        sql.Identifier(settings.db_name)
                    )
                )
                print(f"[init_db] Created database: {settings.db_name}")
            else:
                print(f"[init_db] Database exists: {settings.db_name}")
    finally:
        conn.close()


def init_schema() -> None:
    """在目标库执行所有 DDL。"""
    engine = create_engine(settings.db_url, connect_args=settings.db_connect_args)
    with engine.begin() as conn:
        for stmt in DDL_STATEMENTS:
            conn.execute(text(stmt))
    print(f"[init_db] Schema initialized ({len(DDL_STATEMENTS)} statements applied)")


def main() -> None:
    print(f"[init_db] Target: {settings.db_host}:{settings.db_port}/{settings.db_name}")
    ensure_database()
    init_schema()

    # 简单自检：列出目标库的所有表
    engine = create_engine(settings.db_url, connect_args=settings.db_connect_args)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname='public' ORDER BY tablename"
            )
        ).fetchall()
    print("[init_db] Tables in public schema:")
    for (name,) in rows:
        print(f"    - {name}")


if __name__ == "__main__":
    main()
