"""把 3 个 CSV 灌进 PostgreSQL。

策略：
- ad_feature   全量导入（~85 万行）
- user_profile 全量导入（~106 万行）
- raw_sample   按 clk 列**分层采样** RAW_SAMPLE_SIZE 行（默认 30 万），保证正负样本比与原始一致

幂等：每次执行先 TRUNCATE 三张业务表再灌入。

用法：
    python scripts/load_data.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# 让脚本能 import app.*
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.config import settings


RANDOM_SEED = 42

# PostgreSQL 单条 SQL 绑定参数上限 65535 (int16)。
# pandas.to_sql(method="multi") 会把 chunksize × n_cols 个参数拼进一条 INSERT，
# 必须算出每次插入多少行才不会超限。留余量取 60000。
_PG_SAFE_PARAMS = 60_000


def chunksize_for(n_cols: int) -> int:
    """根据列数返回安全的 chunksize。"""
    return max(1, _PG_SAFE_PARAMS // n_cols)


def make_engine() -> Engine:
    return create_engine(settings.db_url, connect_args=settings.db_connect_args)


def truncate_business_tables(engine: Engine) -> None:
    """清空 3 张业务表（重置 raw_sample 的 SERIAL 计数）。"""
    with engine.begin() as conn:
        conn.execute(
            text("TRUNCATE TABLE ad_feature, raw_sample, user_profile RESTART IDENTITY")
        )
    print("[load_data] Truncated business tables")


def load_ad_feature(engine: Engine, csv_path: Path) -> int:
    """全量导入 ad_feature。"""
    t0 = time.time()
    df = pd.read_csv(csv_path)
    df.to_sql(
        "ad_feature",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=chunksize_for(len(df.columns)),
    )
    elapsed = time.time() - t0
    print(f"[load_data] ad_feature: {len(df):>9,} rows in {elapsed:.1f}s")
    return len(df)


def load_user_profile(engine: Engine, csv_path: Path) -> int:
    """全量导入 user_profile。CSV 表头最后一列含尾随空格，做 strip。"""
    t0 = time.time()
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df.to_sql(
        "user_profile",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=chunksize_for(len(df.columns)),
    )
    elapsed = time.time() - t0
    print(f"[load_data] user_profile: {len(df):>9,} rows in {elapsed:.1f}s")
    return len(df)


def load_raw_sample_stratified(
    engine: Engine, csv_path: Path, sample_size: int
) -> tuple[int, dict[int, int]]:
    """按 clk 分层采样后导入 raw_sample。

    返回 (采样行数, {clk_value: count}) 用于验证比例。
    """
    t0 = time.time()

    # 一次性读取（~1.2 GB，应届生开发机内存通常够）。
    # 若内存吃紧可改为 chunked + 每个 chunk 按相同 frac 采样后拼接。
    print("[load_data] reading raw_sample.csv (may take ~1 min)...")
    df = pd.read_csv(csv_path)

    total = len(df)
    frac = sample_size / total
    print(
        f"[load_data] raw_sample total={total:,}, "
        f"target={sample_size:,}, frac={frac:.4%}"
    )

    # 按 clk 列做分层抽样：每组同比例采样
    sampled = (
        df.groupby("clk", group_keys=False)
        .sample(frac=frac, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    # CSV 列 user 对应 DB 列 user_id（user 在 PG 是关键字）
    sampled = sampled.rename(columns={"user": "user_id"})
    # 不写 BIGSERIAL 主键 id，让 PG 自增
    sampled.to_sql(
        "raw_sample",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=chunksize_for(len(sampled.columns)),
    )

    clk_counts = sampled["clk"].value_counts().to_dict()
    elapsed = time.time() - t0
    print(
        f"[load_data] raw_sample: {len(sampled):>9,} rows in {elapsed:.1f}s "
        f"(clk distribution: {clk_counts})"
    )
    return len(sampled), clk_counts


def verify(engine: Engine) -> None:
    """读回 COUNT 与 clk 占比，确认数据已落库。"""
    with engine.connect() as conn:
        ad_n = conn.execute(text("SELECT COUNT(*) FROM ad_feature")).scalar_one()
        up_n = conn.execute(text("SELECT COUNT(*) FROM user_profile")).scalar_one()
        rs_n = conn.execute(text("SELECT COUNT(*) FROM raw_sample")).scalar_one()
        clk_dist = conn.execute(
            text(
                "SELECT clk, COUNT(*) AS n FROM raw_sample "
                "GROUP BY clk ORDER BY clk"
            )
        ).all()
    print("\n[verify] Row counts in DB:")
    print(f"    ad_feature   = {ad_n:>9,}")
    print(f"    user_profile = {up_n:>9,}")
    print(f"    raw_sample   = {rs_n:>9,}")
    print("[verify] raw_sample clk distribution:")
    for clk, n in clk_dist:
        pct = n / rs_n * 100
        print(f"    clk={clk}: {n:>7,} ({pct:.2f}%)")


def main() -> None:
    data_dir = Path(settings.raw_data_dir)
    if not data_dir.exists():
        sys.exit(f"[load_data] ERROR: RAW_DATA_DIR does not exist: {data_dir}")

    files = {
        "ad_feature": data_dir / "ad_feature.csv",
        "user_profile": data_dir / "user_profile.csv",
        "raw_sample": data_dir / "raw_sample.csv",
    }
    for name, path in files.items():
        if not path.exists():
            sys.exit(f"[load_data] ERROR: missing CSV file: {path}")

    print(f"[load_data] Source: {data_dir}")
    print(f"[load_data] Target: {settings.db_host}:{settings.db_port}/{settings.db_name}")
    print(f"[load_data] raw_sample target size: {settings.raw_sample_size:,}")

    engine = make_engine()
    truncate_business_tables(engine)

    load_ad_feature(engine, files["ad_feature"])
    load_user_profile(engine, files["user_profile"])
    load_raw_sample_stratified(engine, files["raw_sample"], settings.raw_sample_size)

    verify(engine)
    print("\n[load_data] DONE.")


if __name__ == "__main__":
    main()
