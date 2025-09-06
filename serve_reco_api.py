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
from functools import lru_cache
import logging
import time

# ====== Config Logging ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== Config ======
OUTDIR = os.getenv("HR_OUTDIR", "model_out")
CLEAN_PATH = os.path.join(OUTDIR, "cleaned_hr.csv")
MODEL_PATH = os.path.join(OUTDIR, "rf_bal_model.joblib")
PREP_PATH = os.path.join(OUTDIR, "preprocess_bal.joblib")

# ====== Load artifacts ======
if not (os.path.exists(CLEAN_PATH) and os.path.exists(MODEL_PATH) and os.path.exists(PREP_PATH)):
    raise RuntimeError("Thiếu artifacts. Hãy train trước bằng: python hr_reco.py train --input processed_hr.csv --outdir model_out")

clean = pd.read_csv(CLEAN_PATH)
rf = joblib.load(MODEL_PATH)
preprocess = joblib.load(PREP_PATH)

FEATURE_COLS = [
    "hr_score", "bod_score", "kpi_2024", "kpi_2025",
    "tenure_months", "compliance_score", "overall_score",
    "role", "store_kpi24_mean", "store_violations_mean"
]

# Precompute medians
MEDIANS = {}
for col in ['hr_score', 'bod_score', 'kpi_2024', 'kpi_2025', 'tenure_months', 'compliance_score', 'overall_score']:
    MEDIANS[col] = clean[col].median(skipna=True)

def _build_feats_for_target_store(df_employees: pd.DataFrame, target_store_row: pd.Series) -> pd.DataFrame:
    start_time = time.time()
    feats = df_employees[['hr_score', 'bod_score', 'kpi_2024', 'kpi_2025',
                          'tenure_months', 'compliance_score', 'overall_score', 'role']].copy()
    feats['store_kpi24_mean'] = target_store_row.get('store_kpi24_mean', np.nan)
    feats['store_violations_mean'] = target_store_row.get('store_violations_mean', np.nan)

    num_cols = feats.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        feats[col] = pd.to_numeric(feats[col], errors='coerce').fillna(MEDIANS.get(col, 0.0))

    feats['role'] = feats['role'].fillna('Unknown')
    logger.info(f"Feature building took {time.time() - start_time:.2f}s for {len(feats)} candidates")
    return feats

@lru_cache(maxsize=512)
def _recommend_for_store(store_id: int, top_n: int = 5, exclude_same_store: bool = True) -> pd.DataFrame:
    start_time = time.time()
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

    keep_cols = ['store_id', 'store_name', 'employee_id', 'employee_name', 'role',
                 'overall_score', 'violations_total', 'kpi_2024', 'tenure_months']
    keep_cols = [c for c in keep_cols if c in candidates.columns]
    out = candidates[keep_cols].copy()
    out['target_store_id'] = store_id
    out['success_proba'] = proba
    out = out.sort_values('success_proba', ascending=False).head(top_n).reset_index(drop=True)
    
    logger.info(f"Recommendation for store_id={store_id}, top_n={top_n} took {time.time() - start_time:.2f}s")
    return out

@lru_cache(maxsize=512)
def _rank_employees(top_n: int, intent: str = "highest_score") -> pd.DataFrame:
    start_time = time.time()
    ranking_col = 'overall_score'
    if intent == "lowest_score":
        ranked = clean.sort_values(ranking_col, ascending=True).head(top_n)
    else:  # highest_score or default
        ranked = clean.sort_values(ranking_col, ascending=False).head(top_n)

    keep_cols = ['store_id', 'store_name', 'employee_id', 'employee_name', 'role',
                 'overall_score', 'violations_total', 'kpi_2024', 'tenure_months']
    keep_cols = [c for c in keep_cols if c in ranked.columns]
    out = ranked[keep_cols].copy()
    out['rank_metric'] = ranked[ranking_col]

    logger.info(f"Ranking {top_n} employees by {intent} took {time.time() - start_time:.2f}s")
    return out

def _list_all_employees(limit: int = 1000) -> pd.DataFrame:
    start_time = time.time()
    keep_cols = ['store_id', 'store_name', 'employee_id', 'employee_name', 'role',
                 'overall_score', 'violations_total', 'kpi_2024', 'tenure_months']
    keep_cols = [c for c in keep_cols if c in clean.columns]
    out = clean[keep_cols].copy().head(limit)  # Limit to avoid overwhelming response
    logger.info(f"Listed {len(out)} employees took {time.time() - start_time:.2f}s")
    return out

# ====== FastAPI ======
app = FastAPI(title="HR Transfer Recommendation API")

class RecommendRequest(BaseModel):
    store_id: int
    top: int = 5
    exclude_same_store: bool = True

class RankRequest(BaseModel):
    top: int = 5
    intent: str = "highest_score"

class ListAllRequest(BaseModel):
    limit: int = 1000

@app.get("/stores")
def list_stores(limit: int = Query(200, ge=1, le=1000)):
    df = clean[['store_id', 'store_name']].drop_duplicates().sort_values('store_id').head(limit)
    return df.to_dict(orient='records')

@app.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        rec = _recommend_for_store(req.store_id, req.top, req.exclude_same_store)
        data = json.loads(rec.to_json(orient='records'))
        return {"store_id": req.store_id, "top": req.top, "data": data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rank")
def rank(req: RankRequest):
    try:
        rec = _rank_employees(req.top, req.intent)
        data = json.loads(rec.to_json(orient='records'))
        return {"top": req.top, "intent": req.intent, "data": data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/list_all")
def list_all(req: ListAllRequest = None):
    try:
        data = json.loads(_list_all_employees(req.limit if req else 1000).to_json(orient='records'))
        return {"limit": req.limit if req else 1000, "data": data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))