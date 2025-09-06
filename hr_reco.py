#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HR Transfer Recommendation — VS Code runnable
- Clean CSV
- Feature engineering
- SMOTE-balanced training
- Save model & preprocessing
- Recommend Top-N candidates for a target store

Usage:
  python hr_reco.py train --input processed_hr.csv --outdir model_out
  python hr_reco.py recommend --store-id 270 --outdir model_out --top 5
"""

import argparse
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------
# Cleaning helpers
# ---------------------------

ROLE_MAP = {
    "quan ly": "Quản lý",
    "quản lý": "Quản lý",
    "pho quan ly": "Phó quản lý",
    "phó quản lý": "Phó quản lý",
    "quan ly tap su": "Quản lý tập sự",
    "quản lý tập sự": "Quản lý tập sự",
    "ql": "Quản lý",
    "pql": "Phó quản lý",
}
DATE_COL = "appointed_date"
NUM_COLS = [
    "hr_score",
    "bod_score",
    "kpi_2024",
    "kpi_2025",
    "violations_2022",
    "violations_2023",
    "violations_2024",
]


def _strip_collapse(s: Optional[str]) -> Optional[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    return re.sub(r"\s+", " ", str(s)).strip()


def _normalize_role(x: Optional[str]) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = _strip_collapse(x)
    s_flat = re.sub(r"\s+", " ", s).lower()
    return ROLE_MAP.get(s_flat, s)


def _coerce_int(x):
    try:
        return int(str(x).strip())
    except Exception:
        return np.nan


def _parse_date_mixed(s: Optional[str]) -> Optional[pd.Timestamp]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return pd.NaT
    s = str(s).strip()
    if s == "" or s.lower() in ["0", "na", "none", "null"]:
        return pd.NaT
    s = s.replace(".", "-").replace("/", "-")
    for dayfirst in (True, False):
        try:
            return pd.to_datetime(s, dayfirst=dayfirst, errors="raise")
        except Exception:
            continue
    return pd.NaT


def _tenure_months(dt: Optional[pd.Timestamp], ref: Optional[pd.Timestamp] = None) -> float:
    if pd.isna(dt):
        return np.nan
    if ref is None:
        ref = pd.Timestamp.today().normalize()
    return (ref.year - dt.year) * 12 + (ref.month - dt.month) - (0 if ref.day >= dt.day else 1)


def clean_hr_dataframe(df: pd.DataFrame, min_mature_days: int = 90) -> pd.DataFrame:
    # Trim text
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(_strip_collapse)

    # IDs
    if "store_id" in df.columns:
        df["store_id"] = df["store_id"].apply(_coerce_int)
    if "employee_id" in df.columns:
        df["employee_id"] = df["employee_id"].astype(str).apply(_strip_collapse)

    # Role
    if "role" in df.columns:
        df["role"] = df["role"].apply(_normalize_role)

    # Dates
    if DATE_COL in df.columns:
        ts = df[DATE_COL].apply(_parse_date_mixed)
        df[DATE_COL] = ts.dt.date.astype("string")
        df["_appointed_ts"] = ts

    # Numeric
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(",", ".", regex=False)
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Violations total
    if all(c in df.columns for c in ["violations_2022", "violations_2023", "violations_2024"]):
        df["violations_total"] = (
            df[["violations_2022", "violations_2023", "violations_2024"]].sum(axis=1, skipna=True)
        )
    else:
        df["violations_total"] = np.nan

    # Tenure
    if "_appointed_ts" in df.columns:
        ref = pd.Timestamp.today().normalize()
        df["tenure_months"] = df["_appointed_ts"].apply(lambda d: _tenure_months(d, ref))
    else:
        df["tenure_months"] = np.nan

    # Zero-as-missing for mature non-trainees
    if {"hr_score", "bod_score", "kpi_2024"}.issubset(df.columns):
        mature = (df["tenure_months"] >= min_mature_days / 30.0) & (df["role"] != "Quản lý tập sự")
        for c in ["hr_score", "bod_score", "kpi_2024", "kpi_2025"]:
            if c in df.columns:
                df.loc[mature & (df[c] == 0), c] = np.nan

    # Compliance score
    w = {"violations_2022": 5, "violations_2023": 7, "violations_2024": 10}
    if all(c in df.columns for c in w):
        comp = 100
        for c, wt in w.items():
            comp = comp - wt * df[c].fillna(0)
        df["compliance_score"] = comp.clip(0, 100)
    else:
        df["compliance_score"] = np.nan

    # Overall score (0–100 scale)
    def _overall(r):
        parts = []
        if not pd.isna(r.get("hr_score")):
            parts.append(0.35 * r["hr_score"])
        if not pd.isna(r.get("bod_score")):
            parts.append(0.35 * r["bod_score"])
        if not pd.isna(r.get("kpi_2024")):
            parts.append(0.20 * (r["kpi_2024"] * 25.0))  # scale KPI ~2.x to 0–100
        if not pd.isna(r.get("compliance_score")):
            parts.append(0.10 * (r["compliance_score"]))
        return np.nan if not parts else round(sum(parts), 2)

    df["overall_score"] = df.apply(_overall, axis=1)

    # Store-level benchmarks
    if "store_id" in df.columns and "kpi_2024" in df.columns:
        bench = (
            df.groupby("store_id", dropna=False)
            .agg(store_kpi24_mean=("kpi_2024", "mean"), store_violations_mean=("violations_total", "mean"))
            .reset_index()
        )
        df = df.merge(bench, on="store_id", how="left")

    # Dedup newest per (store_id, employee_id)
    if {"store_id", "employee_id", "_appointed_ts"}.issubset(df.columns):
        df = df.sort_values(["store_id", "employee_id", "_appointed_ts"], ascending=[True, True, False])
        df = df.drop_duplicates(subset=["store_id", "employee_id"], keep="first")

    if "_appointed_ts" in df.columns:
        df = df.drop(columns=["_appointed_ts"])

    # Column order (best effort)
    preferred = [
        "store_id",
        "store_name",
        "employee_id",
        "employee_name",
        "role",
        "appointed_date",
        "hr_score",
        "bod_score",
        "kpi_2024",
        "kpi_2025",
        "violations_2022",
        "violations_2023",
        "violations_2024",
        "violations_total",
        "tenure_months",
        "compliance_score",
        "overall_score",
        "store_kpi24_mean",
        "store_violations_mean",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


# ---------------------------
# Modeling
# ---------------------------

FEATURE_COLS = [
    "hr_score",
    "bod_score",
    "kpi_2024",
    "kpi_2025",
    "tenure_months",
    "compliance_score",
    "overall_score",
    "role",
    "store_kpi24_mean",
    "store_violations_mean",
]


@dataclass
class TrainArtifacts:
    clean: pd.DataFrame
    preprocess: ColumnTransformer
    rf_model: RandomForestClassifier


def _build_df_model(clean: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    dfm = clean.copy()
    # Rule-based label (có thể chỉnh)
    if "label" not in dfm.columns:
        dfm["label"] = np.where((dfm["overall_score"] >= 70) & (dfm["violations_total"] <= 3), 1, 0)

    # Impute đơn giản như lúc train notebook
    for c in [x for x in FEATURE_COLS if x != "role"]:
        if c in dfm.columns:
            dfm[c] = pd.to_numeric(dfm[c], errors="coerce")
            dfm[c] = dfm[c].fillna(dfm[c].median())
    if "role" in dfm.columns:
        dfm["role"] = dfm["role"].fillna("Unknown")

    X = dfm[FEATURE_COLS].copy()
    y = dfm["label"].astype(int)
    return X, y


def _fit_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_features = [c for c in X.columns if c != "role"]
    cat_features = ["role"]
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )
    preprocess.fit(X)
    return preprocess


def _fit_balanced_model(X: pd.DataFrame, y: pd.Series, preprocess: ColumnTransformer) -> RandomForestClassifier:
    X_t = preprocess.transform(X)

    # Xử lý mất cân bằng: SMOTE (an toàn) hoặc RandomOverSampler fallback
    classes, counts = np.unique(y, return_counts=True)
    minority_count = counts.min()
    k_neighbors = max(1, min(5, minority_count - 1))  # đảm bảo < minority_count
    if minority_count >= 2:
        try:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_res, y_res = smote.fit_resample(X_t, y)
        except Exception:
            ros = RandomOverSampler(random_state=42)
            X_res, y_res = ros.fit_resample(X_t, y)
    else:
        # quá ít mẫu dương/âm → không resample được
        X_res, y_res = X_t, y

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.3, random_state=42, stratify=y_res if len(np.unique(y_res)) > 1 else None
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",  # tăng độ bền khi còn lệch lớp
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    print("=== Evaluation (balanced set) ===")
    if len(np.unique(y_test)) > 1:
        print(classification_report(y_test, y_pred, zero_division=0))
        try:
            print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))
        except Exception:
            pass
    else:
        print("Only one class in test split; metrics limited.")

    return rf


def train(input_csv: str, outdir: str) -> TrainArtifacts:
    os.makedirs(outdir, exist_ok=True)
    raw = pd.read_csv(input_csv, dtype=str)
    clean = clean_hr_dataframe(raw, min_mature_days=90)
    clean_path = os.path.join(outdir, "cleaned_hr.csv")
    clean.to_csv(clean_path, index=False, encoding="utf-8")
    print(f"[OK] cleaned → {clean_path} rows={len(clean)}")

    X, y = _build_df_model(clean)
    preprocess = _fit_preprocessor(X)
    rf = _fit_balanced_model(X, y, preprocess)

    # Lưu artifacts
    joblib.dump(rf, os.path.join(outdir, "rf_bal_model.joblib"))
    joblib.dump(preprocess, os.path.join(outdir, "preprocess_bal.joblib"))
    print(f"[OK] saved model & preprocessing → {outdir}")

    # Demo: tạo gợi ý cho 3 cửa hàng KPI thấp nhất
    store_stats = clean[["store_id", "store_name", "store_kpi24_mean", "store_violations_mean"]].drop_duplicates()
    challenged = (
        store_stats.dropna(subset=["store_kpi24_mean"])
        .sort_values(["store_kpi24_mean", "store_violations_mean"])
        .head(3)
    )
    all_recs = []
    for _, row in challenged.iterrows():
        sid = int(row["store_id"]) if not pd.isna(row["store_id"]) else None
        rec = recommend_for_store(sid, clean, preprocess, rf, top_n=5)
        rec["target_store_name"] = row["store_name"]
        all_recs.append(rec)
    if all_recs:
        recs_df = pd.concat(all_recs, ignore_index=True)
        recs_path = os.path.join(outdir, "recommendations_demo.csv")
        recs_df.to_csv(recs_path, index=False, encoding="utf-8")
        print(f"[OK] demo recommendations → {recs_path}")

    return TrainArtifacts(clean=clean, preprocess=preprocess, rf_model=rf)


# ---------------------------
# Recommendation
# ---------------------------

def _build_feats_for_target_store(df_employees: pd.DataFrame, target_store_row: pd.Series) -> pd.DataFrame:
    feats = df_employees[
        ["hr_score", "bod_score", "kpi_2024", "kpi_2025", "tenure_months", "compliance_score", "overall_score", "role"]
    ].copy()
    feats["store_kpi24_mean"] = target_store_row.get("store_kpi24_mean", np.nan)
    feats["store_violations_mean"] = target_store_row.get("store_violations_mean", np.nan)

    # same imputation as training
    for c in ["hr_score", "bod_score", "kpi_2024", "kpi_2025", "tenure_months", "compliance_score", "overall_score",
              "store_kpi24_mean", "store_violations_mean"]:
        feats[c] = pd.to_numeric(feats[c], errors="coerce").fillna(df_employees[c].median(skipna=True))
    feats["role"] = feats["role"].fillna("Unknown")
    return feats


def recommend_for_store(
    store_id: int, clean_df: pd.DataFrame, preprocess: ColumnTransformer, model: RandomForestClassifier, top_n: int = 5
) -> pd.DataFrame:
    target_rows = clean_df[clean_df["store_id"] == store_id]
    if target_rows.empty:
        raise ValueError(f"Store {store_id} not found.")
    target = target_rows.iloc[0]

    candidates = clean_df[clean_df["store_id"] != store_id].copy()
    feats = _build_feats_for_target_store(candidates, target)
    X = preprocess.transform(feats)
    proba = model.predict_proba(X)[:, 1]

    out_cols = [
        "store_id",
        "store_name",
        "employee_id",
        "employee_name",
        "role",
        "overall_score",
        "violations_total",
        "kpi_2024",
        "tenure_months",
    ]
    out_cols = [c for c in out_cols if c in candidates.columns]
    out = candidates[out_cols].copy()
    out["target_store_id"] = store_id
    out["success_proba"] = proba
    out = out.sort_values("success_proba", ascending=False).head(top_n).reset_index(drop=True)
    return out


def cli_recommend(store_id: int, outdir: str, top: int):
    # Load artifacts
    clean = pd.read_csv(os.path.join(outdir, "cleaned_hr.csv"))
    rf = joblib.load(os.path.join(outdir, "rf_bal_model.joblib"))
    preprocess = joblib.load(os.path.join(outdir, "preprocess_bal.joblib"))

    rec = recommend_for_store(store_id, clean, preprocess, rf, top_n=top)
    out_path = os.path.join(outdir, f"recommendations_store_{store_id}.csv")
    rec.to_csv(out_path, index=False, encoding="utf-8")
    print(rec.head(top))
    print(f"[OK] saved recommendations → {out_path}")


# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("train", help="Clean, train, save artifacts")
    a.add_argument("--input", default="processed_hr.csv", help="Path to raw CSV")
    a.add_argument("--outdir", default="model_out", help="Output directory")

    r = sub.add_parser("recommend", help="Recommend candidates for a store")
    r.add_argument("--store-id", type=int, required=True)
    r.add_argument("--outdir", default="model_out")
    r.add_argument("--top", type=int, default=5)

    args = ap.parse_args()

    if args.cmd == "train":
        train(args.input, args.outdir)
    elif args.cmd == "recommend":
        cli_recommend(args.store_id, args.outdir, args.top)


if __name__ == "__main__":
    main()
