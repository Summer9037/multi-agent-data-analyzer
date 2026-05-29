"""清理孤儿运行时产物。

哪些算"孤儿"？
================
- `cleaning_work_<uuid>`: 任何 run 都不该有这种"中间态"表残留
  （persist_cleaned 应该把它 RENAME 掉了；残留意味着 cleaner 节点中途崩了）
- `cleaned_<uuid>` / `feature_wide_<uuid>`: 如果 uuid 在 analysis_runs 里
  status='success' 就保留；其他都删
- `outputs/reports/<uuid>.md` / `outputs/charts/<uuid>/`：同上

用法
====
    python scripts/cleanup_stale_tables.py            # 干跑（只打印不删）
    python scripts/cleanup_stale_tables.py --apply    # 实际执行

为什么默认 dry-run
==================
删表 / 删文件不可逆。先看一遍要删什么，确认后再加 --apply。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.config import settings
from app.tools.db import get_engine


# ============ DB 部分 ============

def _list_runtime_tables(conn) -> list[str]:
    """返回所有 cleaning_work_* / cleaned_* / feature_wide_* 表名。"""
    rows = conn.execute(
        text(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname = 'public' "
            "AND (tablename LIKE 'cleaning_work_%' "
            "     OR tablename LIKE 'cleaned_%' "
            "     OR tablename LIKE 'feature_wide_%') "
            "ORDER BY tablename"
        )
    ).fetchall()
    return [r[0] for r in rows]


def _success_run_ids(conn) -> set[str]:
    """analysis_runs 里 status='success' 的 run_id（连字符版）。"""
    rows = conn.execute(
        text("SELECT run_id FROM analysis_runs WHERE status = 'success'")
    ).fetchall()
    return {str(r[0]) for r in rows}


def _table_run_id(tbl: str) -> str | None:
    """从表名抽出 run_id（下划线版 → 连字符版），失败返回 None。"""
    for prefix in ("cleaning_work_", "feature_wide_", "cleaned_"):
        if tbl.startswith(prefix):
            uuid_part = tbl[len(prefix):]
            return uuid_part.replace("_", "-")
    return None


def cleanup_db_tables(apply: bool) -> tuple[list[str], list[str]]:
    """返回 (kept_tables, dropped_tables)。"""
    eng = get_engine()
    kept: list[str] = []
    dropped: list[str] = []

    with eng.begin() as conn:
        tables = _list_runtime_tables(conn)
        if not tables:
            print("[db] 无运行时表，无需清理")
            return [], []

        success_ids = _success_run_ids(conn)

        for tbl in tables:
            rid = _table_run_id(tbl)
            # cleaning_work_* 一律删（无论 run 是否成功，这都是中间态）
            if tbl.startswith("cleaning_work_"):
                dropped.append(tbl)
                if apply:
                    conn.execute(text(f"DROP TABLE IF EXISTS {tbl}"))
                continue

            # cleaned_* / feature_wide_*：留下 status='success' 的
            if rid and rid in success_ids:
                kept.append(tbl)
            else:
                dropped.append(tbl)
                if apply:
                    conn.execute(text(f"DROP TABLE IF EXISTS {tbl}"))

    return kept, dropped


# ============ 文件系统部分 ============

def cleanup_output_files(apply: bool) -> tuple[list[str], list[str]]:
    """outputs/reports/<run_id>.md + outputs/charts/<run_id>/ 同样按 run_id 比对。"""
    out_dir = Path(settings.output_dir)
    if not out_dir.exists():
        return [], []

    with get_engine().connect() as conn:
        success_ids = _success_run_ids(conn)

    kept: list[str] = []
    dropped: list[str] = []

    # 1) reports
    reports = (out_dir / "reports")
    if reports.exists():
        for f in reports.glob("*.md"):
            rid = f.stem
            if rid in success_ids:
                kept.append(str(f))
            else:
                dropped.append(str(f))
                if apply:
                    f.unlink()

    # 2) charts
    charts = (out_dir / "charts")
    if charts.exists():
        for d in charts.iterdir():
            if not d.is_dir():
                continue
            rid = d.name
            if rid in success_ids:
                kept.append(str(d))
            else:
                dropped.append(str(d))
                if apply:
                    shutil.rmtree(d, ignore_errors=True)

    return kept, dropped


# ============ 主入口 ============

def main() -> None:
    parser = argparse.ArgumentParser(description="清理孤儿运行时表与产物")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="实际执行删除（默认仅打印 dry-run 结果）",
    )
    args = parser.parse_args()

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== Cleanup [{mode}] ===\n")

    print("--- DB tables ---")
    kept_t, dropped_t = cleanup_db_tables(apply=args.apply)
    print(f"  保留: {len(kept_t)}")
    for t in kept_t:
        print(f"    KEEP  {t}")
    print(f"  {'已删除' if args.apply else '将删除'}: {len(dropped_t)}")
    for t in dropped_t:
        print(f"    DROP  {t}")

    print()
    print("--- 输出文件 ---")
    kept_f, dropped_f = cleanup_output_files(apply=args.apply)
    print(f"  保留: {len(kept_f)}")
    for p in kept_f:
        print(f"    KEEP  {p}")
    print(f"  {'已删除' if args.apply else '将删除'}: {len(dropped_f)}")
    for p in dropped_f:
        print(f"    DROP  {p}")

    if not args.apply:
        print("\n*** Dry-run 模式 *** 加 --apply 才会真正执行删除。")


if __name__ == "__main__":
    main()
