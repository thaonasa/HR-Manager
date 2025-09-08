#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
from functools import lru_cache
import logging
import time
import re

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

if 'store_id' not in clean.columns:
    # Nếu index đang là kiểu số và có vẻ là store_id
    if (clean.index.name in (None, 'store_id')) and pd.api.types.is_integer_dtype(clean.index):
        clean = clean.reset_index().rename(columns={'index': 'store_id' if 'store_id' not in clean.columns else 'index'})
    else:
        # Một số pipeline có thể ghi nhầm tên cột
        possible = [c for c in clean.columns if c.lower().replace('-', '').replace(' ', '') in ('storeid','store_id')]
        if possible:
            clean = clean.rename(columns={possible[0]: 'store_id'})

# 2) Ép kiểu số nguyên nullable để to_json không rơi mất cột
if 'store_id' in clean.columns:
    clean['store_id'] = pd.to_numeric(clean['store_id'], errors='coerce').astype('Int64')
else:
    # Nếu bất khả kháng, thêm cột null để frontend vẫn có key
    clean['store_id'] = pd.Series([pd.NA]*len(clean), dtype='Int64')

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
        logger.warning(f"Không tìm thấy store_id={store_id} trong dataset")
        return pd.DataFrame()  # Trả về DataFrame rỗng nếu không tìm thấy

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
    out = clean[keep_cols].copy().head(limit)
    logger.info(f"Listed {len(out)} employees took {time.time() - start_time:.2f}s")
    return out

# ====== FastAPI ======
app = FastAPI(title="HR Transfer Recommendation API")

# Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5000", "http://localhost:5000"],  # Cho phép origin của Flask
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả phương thức
    allow_headers=["*"],
)

class RecommendRequest(BaseModel):
    store_id: int
    top: int = 5
    exclude_same_store: bool = True

class RankRequest(BaseModel):
    top: int = 5
    intent: str = "highest_score"

class ListAllRequest(BaseModel):
    limit: int = 1000

class ChatRequest(BaseModel):
    message: str
    session_id: str = "session-1"

@app.get("/stores")
def list_stores(limit: int = Query(200, ge=1, le=1000)):
    df = clean[['store_id', 'store_name']].drop_duplicates().sort_values('store_id').head(limit)
    return df.to_dict(orient='records')

@app.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        rec = _recommend_for_store(req.store_id, req.top, req.exclude_same_store)
        if rec.empty:
            raise ValueError(f"Không tìm thấy dữ liệu cho store_id={req.store_id}")
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
def list_all(limit: int = 200):
    try:
        data = json.loads(_list_all_employees(limit).to_json(orient='records'))
        return {
            "type": "employees_list",
            "payload": {"employees": data},
            "meta": {"limit": limit, "title": f"Danh sách toàn bộ nhân sự ({limit} dòng đầu)"},
            "response": "Danh sách toàn bộ nhân sự"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Trả về JSON có cấu trúc:
    - Khi list all:    {"type":"employees_list","payload":{"employees":[...]}, "meta":{...}, "response":"..."}
    - Khi recommend:   {"type":"employees_list","payload":{"employees":[...]}, "meta":{"store_id":..., "top":...}, "response":"..."}
    - Khi rank:        {"type":"rank_list","payload":{"employees":[...]}, "meta":{"top":..., "intent":...}, "response":"..."}
    - Fallback (text): {"response":"..."}
    """
    try:
        message = (req.message or "").lower()
        session_id = req.session_id

        # Bắt tham số
        m_store = re.search(r"(cho\s+(cửa hàng|store))\s*(\d+)", message)
        store_id = int(m_store.group(3)) if m_store else None

        m_top = re.search(r"(top|đề xuất|gợi ý)\s*(\d+)", message)
        top = int(m_top.group(2)) if m_top else 3

        # Xác định intent
        intent = "recommend"
        if re.search(r"chỉ số thấp nhất|thấp nhất", message):
            intent = "lowest_score"
        elif re.search(r"điểm cao nhất|cao nhất", message):
            intent = "highest_score"
        elif re.search(r"(show\s+toàn bộ nhân sự|danh sách nhân sự|hiển thị tất cả|show all)", message):
            intent = "list_all"

        # Suy luận store_id nếu thiếu
        if store_id is None and intent == "recommend":
            nums = re.findall(r"\d+", message)
            store_id = int(nums[-1]) if nums else None

        # Xử lý các intent
        if intent == "recommend" and store_id is not None:
            store_rows = clean[clean['store_id'] == store_id]
            if store_rows.empty:
                return {
                    "response": f"⚠️ Không tìm thấy cửa hàng {store_id}. "
                                f"Hãy thử một trong các store: {sorted(clean['store_id'].dropna().unique().tolist())[:50]}..."
                }

            df = _recommend_for_store(store_id, top, True)
            employees = json.loads(df.to_json(orient='records'))
            return {
                "type": "employees_list",
                "payload": {"employees": employees},
                "meta": {
                    "store_id": store_id,
                    "top": top,
                    "title": f"Top {top} ứng viên cho cửa hàng {store_id}"
                },
                "response": f"Top {top} ứng viên cho cửa hàng {store_id}"
            }

        elif intent in ["lowest_score", "highest_score"]:
            df = _rank_employees(top, intent)
            employees = json.loads(df.to_json(orient='records'))
            title = f"Top {top} người {'có chỉ số thấp nhất' if intent == 'lowest_score' else 'có điểm cao nhất'}"
            return {
                "type": "rank_list",
                "payload": {"employees": employees},
                "meta": {"top": top, "intent": intent, "title": title},
                "response": title
            }

        elif intent == "list_all":
            df = _list_all_employees(200)  # tránh trả quá nhiều
            employees = json.loads(df.to_json(orient='records'))
            return {
                "type": "employees_list",
                "payload": {"employees": employees},
                "meta": {"limit": 200, "title": "Danh sách toàn bộ nhân sự (200 dòng đầu)"},
                "response": "Danh sách toàn bộ nhân sự"
            }

        else:
            return {
                "response": "Truy vấn không hợp lệ. Hãy thử: "
                            "'gợi ý cho cửa hàng X', 'top N người có chỉ số thấp nhất', "
                            "hoặc 'show toàn bộ nhân sự'."
            }

    except Exception as e:
        logger.exception("Chat endpoint error")
        return {"response": f"⚠️ Lỗi: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)