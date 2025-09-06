#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import json


# ====== Config ======
OUTDIR = os.getenv("HR_OUTDIR", "model_out")  # thư mục chứa artifacts
CLEAN_PATH = os.path.join(OUTDIR, "cleaned_hr.csv")
MODEL_PATH = os.path.join(OUTDIR, "rf_bal_model.joblib")
PREP_PATH  = os.path.join(OUTDIR, "preprocess_bal.joblib")

# ====== Load artifacts ======
if not (os.path.exists(CLEAN_PATH) and os.path.exists(MODEL_PATH) and os.path.exists(PREP_PATH)):
    raise RuntimeError("Thiếu artifacts. Hãy train trước bằng: python hr_reco.py train --input processed_hr.csv --outdir model_out")

clean = pd.read_csv(CLEAN_PATH)
rf = joblib.load(MODEL_PATH)
preprocess = joblib.load(PREP_PATH)

FEATURE_COLS = [
    "hr_score","bod_score","kpi_2024","kpi_2025",
    "tenure_months","compliance_score","overall_score",
    "role","store_kpi24_mean","store_violations_mean"
]

def _build_feats_for_target_store(df_employees: pd.DataFrame, target_store_row: pd.Series) -> pd.DataFrame:
    feats = df_employees[['hr_score','bod_score','kpi_2024','kpi_2025',
                          'tenure_months','compliance_score','overall_score','role']].copy()
    feats['store_kpi24_mean'] = target_store_row.get('store_kpi24_mean', np.nan)
    feats['store_violations_mean'] = target_store_row.get('store_violations_mean', np.nan)

    # Impute số bằng median an toàn (nếu median NaN -> dùng 0)
    num_cols = feats.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        feats[col] = pd.to_numeric(feats[col], errors='coerce')
        med = pd.to_numeric(df_employees[col], errors='coerce').median()
        if pd.isna(med):
            med = 0.0
        feats[col] = feats[col].fillna(med)

    feats['role'] = feats['role'].fillna('Unknown')
    return feats


def _recommend_for_store(store_id: int, top_n: int = 5, exclude_same_store: bool = True) -> pd.DataFrame:
    store_rows = clean[clean['store_id'] == store_id]
    if store_rows.empty:
        raise ValueError(f"Không tìm thấy store_id={store_id}")
    target = store_rows.iloc[0]

    candidates = clean.copy()
    if exclude_same_store:
        candidates = candidates[candidates['store_id'] != store_id]

    feats = _build_feats_for_target_store(candidates, target)
    X = preprocess.transform(feats)
    proba = rf.predict_proba(X)[:, 1]

    keep_cols = ['store_id','store_name','employee_id','employee_name','role',
                 'overall_score','violations_total','kpi_2024','tenure_months']
    keep_cols = [c for c in keep_cols if c in candidates.columns]
    out = candidates[keep_cols].copy()
    out['target_store_id'] = store_id
    out['success_proba'] = proba
    out = out.sort_values('success_proba', ascending=False).head(top_n).reset_index(drop=True)
    return out

# ====== FastAPI ======
app = FastAPI(title="HR Transfer Recommendation API")

class RecommendRequest(BaseModel):
    store_id: int
    top: int = 5
    exclude_same_store: bool = True

@app.get("/stores")
def list_stores(limit: int = Query(200, ge=1, le=1000)):
    df = clean[['store_id','store_name']].drop_duplicates().sort_values('store_id').head(limit)
    return df.to_dict(orient='records')

@app.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        rec = _recommend_for_store(req.store_id, req.top, req.exclude_same_store)
        # CHỐT: chuyển DataFrame -> JSON "null-safe" (NaN -> null)
        data = json.loads(rec.to_json(orient='records'))  # pandas xuất NaN thành null
        return {"store_id": req.store_id, "top": req.top, "data": data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

