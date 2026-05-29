"""建模工具：

工具集（4 个）
==============
- build_feature_table(target_col, use_cleaned_user_table)
    把 raw_sample + ad_feature + user_profile（或 cleaned_<run_id>）JOIN 成
    feature_wide_<run_id>，作为后续训练的统一输入

- train_lr(target_col, test_size)        Logistic Regression
- train_lgbm(target_col, test_size, n_estimators)  LightGBM
    两者都做 stratified split → 训练 → AUC/F1 → 特征重要性，一次性返回

- submit_model_result(...)               Modeler 提交最终结构化结果
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from app.runtime.context import get_run_id_for_sql
from app.tools.db import get_engine

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None  # type: ignore[assignment]


def _feature_table() -> str:
    return f"feature_wide_{get_run_id_for_sql()}"


def _cleaned_user_table() -> str:
    return f"cleaned_{get_run_id_for_sql()}"


def build_feature_table(
    target_col: str = "clk",
    use_cleaned_user_table: bool = True,
) -> dict:
    """JOIN raw_sample + ad_feature + user_profile/cleaned_<run_id>，
    持久化为 feature_wide_<run_id>。

    use_cleaned_user_table=True 时优先使用 cleaned_<run_id>（如果存在），
    否则回退到 user_profile。
    """
    eng = get_engine()
    feat = _feature_table()

    user_table = "user_profile"
    if use_cleaned_user_table:
        cleaned = _cleaned_user_table()
        with eng.connect() as conn:
            exists = conn.execute(
                text(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema='public' AND table_name=:t)"
                ),
                {"t": cleaned},
            ).scalar_one()
        if exists:
            user_table = cleaned

    # 从 time_stamp 派生 hour_of_day (0-23) 与 weekday (0-6) —— CTR 任务里
    # 时间段是强信号（晚高峰 vs 凌晨等）。1494000001 = 2017-05-05 (周五) UTC，
    # 用 PG 的 to_timestamp + extract 直接拿。
    sql_build = f"""
        DROP TABLE IF EXISTS {feat};
        CREATE TABLE {feat} AS
        SELECT
            rs.{target_col} AS target,
            rs.pid,
            rs.time_stamp,
            EXTRACT(HOUR FROM to_timestamp(rs.time_stamp))::INTEGER  AS hour_of_day,
            EXTRACT(DOW  FROM to_timestamp(rs.time_stamp))::INTEGER  AS weekday,
            af.cate_id, af.campaign_id, af.customer, af.brand, af.price,
            up.cms_segid, up.cms_group_id, up.final_gender_code, up.age_level,
            up.pvalue_level, up.shopping_level, up.occupation,
            up.new_user_class_level
        FROM raw_sample rs
        LEFT JOIN ad_feature af ON rs.adgroup_id = af.adgroup_id
        LEFT JOIN {user_table} up ON rs.user_id = up.userid;
    """
    with eng.begin() as conn:
        for stmt in [s.strip() for s in sql_build.split(";") if s.strip()]:
            conn.execute(text(stmt))
        n = conn.execute(text(f"SELECT COUNT(*) FROM {feat}")).scalar_one()
        cols = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name=:t ORDER BY ordinal_position"
            ),
            {"t": feat},
        ).scalars().all()

    return {
        "ok": True,
        "op": "build_feature_table",
        "feature_table": feat,
        "user_source": user_table,
        "n_rows": int(n),
        "columns": list(cols),
        "target_col": "target",
    }


# === 数据加载共享逻辑 ===

def _load_features_df() -> pd.DataFrame:
    """从 feature_wide_<run_id> 读全部数据回内存（30万级别可承受）。"""
    eng = get_engine()
    feat = _feature_table()
    with eng.connect() as conn:
        df = pd.read_sql(text(f"SELECT * FROM {feat}"), conn)
    return df


def _preprocess(df: pd.DataFrame, target_col: str = "target") -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """通用预处理：
    - pid 等字符串列做 label encoding（pandas categorical → codes）
    - 缺失值统一填 -1（缺失本身作为信号，让树模型自然捕获）
    - 返回 (X, y, feature_names)
    """
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col]).copy()

    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category").cat.codes.astype(int)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(-1)
    return X, y, list(X.columns)


def _evaluate(y_true: pd.Series, y_proba: np.ndarray) -> dict:
    """统一的二分类评估指标。"""
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "logloss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    return {k: round(v, 4) for k, v in metrics.items()}


def _top_importance(features: list[str], importances: np.ndarray, top_k: int = 15) -> list[dict]:
    pairs = sorted(zip(features, importances), key=lambda x: -abs(x[1]))[:top_k]
    return [{"feature": f, "importance": round(float(i), 4)} for f, i in pairs]


# === 训练工具 ===

def train_lr(target_col: str = "target", test_size: float = 0.2) -> dict:
    """LogisticRegression：标准化 + L2，stratified split。"""
    df = _load_features_df()
    if target_col not in df.columns:
        return {"error": f"target_col {target_col!r} not in feature table"}

    X, y, feats = _preprocess(df, target_col)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # class_weight='balanced' 应对 CTR 5% 极不平衡：否则 LR 会退化成"全预测 0"
    # 让 precision/recall/f1 都为 0。
    model = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
    model.fit(X_tr_s, y_tr)
    y_proba = model.predict_proba(X_te_s)[:, 1]

    metrics = _evaluate(y_te, y_proba)
    fi = _top_importance(feats, np.abs(model.coef_[0]))
    return {
        "ok": True,
        "op": "train_lr",
        "model_name": "logistic_regression",
        "train_rows": int(len(X_tr)),
        "test_rows": int(len(X_te)),
        "metrics": metrics,
        "feature_importance": fi,
    }


def train_lgbm(
    target_col: str = "target",
    test_size: float = 0.2,
    n_estimators: int = 500,
) -> dict:
    """LightGBM 二分类。

    说明
    ----
    采用朴素 baseline 配置（不做 class_weight / scale_pos_weight / 显式
    categorical_feature）：在 Alibaba 公开 CTR 数据集 8 天裸特征上，这套
    配置实测 AUC 最高（~0.54）。声明 categorical 或加权反而触发早停 / 退化。

    数据本身缺少 user×ad 历史交互特征，公开 benchmark 上限约 0.55-0.60；
    要达到 0.65+ 需要做特征工程（target encoding、用户/广告级历史 CTR
    聚合等），超出 MVP 范围。
    """
    if lgb is None:
        return {"error": "lightgbm not installed"}

    df = _load_features_df()
    if target_col not in df.columns:
        return {"error": f"target_col {target_col!r} not in feature table"}

    X, y, feats = _preprocess(df, target_col)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=int(n_estimators),
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=200,
        max_depth=-1,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    y_proba = model.predict_proba(X_te)[:, 1]

    metrics = _evaluate(y_te, y_proba)
    fi = _top_importance(feats, model.feature_importances_.astype(float))
    return {
        "ok": True,
        "op": "train_lgbm",
        "model_name": "lightgbm",
        "train_rows": int(len(X_tr)),
        "test_rows": int(len(X_te)),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "metrics": metrics,
        "feature_importance": fi,
    }


def submit_model_result(
    model_name: str,
    metrics: dict,
    feature_importance: list[dict],
    notes: str = "",
) -> dict:
    """Modeler 提交结构化的模型结果（同 submit_task_plan / submit_cleaning_report 套路）。"""
    return {
        "ok": True,
        "received": {
            "model_name": str(model_name),
            "metrics": dict(metrics or {}),
            "feature_importance": list(feature_importance or []),
            "notes": str(notes or ""),
        },
    }
