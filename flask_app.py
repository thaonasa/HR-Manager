#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import logging
from typing import Any, Dict

import requests
from flask import Flask, render_template, request, Response, jsonify

# ========================
# Config
# ========================
API_URL = os.getenv("HR_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = float(os.getenv("HR_REQUEST_TIMEOUT", "20"))

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['JSON_AS_ASCII'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hr-flask-gateway")


# ========================
# Helpers
# ========================
def _pass_through(resp: requests.Response) -> Response:
    """
    Trả nguyên status/code/content từ FastAPI về UI.
    """
    headers_to_forward = ("Content-Type", "Cache-Control", "ETag")
    headers = [(k, v) for k, v in resp.headers.items() if k in headers_to_forward]
    return Response(resp.content, status=resp.status_code, headers=headers)


def _api_get(path: str, params: Dict[str, Any] | None = None) -> Response:
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=REQUEST_TIMEOUT)
        return _pass_through(r)
    except requests.RequestException as e:
        logger.exception("GET %s failed", path)
        return jsonify({"response": f"⚠️ Lỗi kết nối đến API: {str(e)}"}), 502


def _api_post(path: str, payload: Dict[str, Any] | None = None) -> Response:
    try:
        r = requests.post(f"{API_URL}{path}", json=payload or {}, timeout=REQUEST_TIMEOUT)
        return _pass_through(r)
    except requests.RequestException as e:
        logger.exception("POST %s failed", path)
        return jsonify({"response": f"⚠️ Lỗi kết nối đến API: {str(e)}"}), 502


def _api_delete(path: str) -> Response:
    try:
        r = requests.delete(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT)
        return _pass_through(r)
    except requests.RequestException as e:
        logger.exception("DELETE %s failed", path)
        return jsonify({"response": f"⚠️ Lỗi kết nối đến API: {str(e)}"}), 502


# ========================
# Routes: UI
# ========================
@app.route('/')
def index():
    return render_template('index.html')


# ========================
# Routes: Proxies to FastAPI
# ========================

# Health check
@app.route('/health', methods=['GET'])
def health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return _pass_through(r)
    except requests.RequestException as e:
        return jsonify({"ok": False, "api_url": API_URL, "error": str(e)}), 502


# ---- Chat ----
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    if not data:
        return jsonify({"response": "⚠️ Không nhận được dữ liệu."}), 400
    return _api_post("/chat", payload=data)


# ---- Sessions CRUD ----
@app.route('/sessions', methods=['GET', 'POST'])
def sessions_root():
    if request.method == 'GET':
        return _api_get("/sessions")
    else:
        payload = request.get_json(silent=True) or {}
        return _api_post("/sessions", payload=payload)

@app.route('/sessions/<sid>/title', methods=['POST'])
def sessions_set_title(sid: str):
    payload = request.get_json(silent=True) or {}
    return _api_post(f"/sessions/{sid}/title", payload=payload)

@app.route('/sessions/<sid>', methods=['DELETE'])
def sessions_delete(sid: str):
    return _api_delete(f"/sessions/{sid}")

@app.route('/sessions/<sid>/messages', methods=['GET'])
def sessions_messages(sid: str):
    params = dict(request.args.items())
    return _api_get(f"/sessions/{sid}/messages", params=params)


# ---- Stores ----
@app.route('/stores', methods=['GET'])
def stores():
    params = dict(request.args.items())
    return _api_get("/stores", params=params)

@app.route('/stores/register', methods=['POST'])
def stores_register():
    payload = request.get_json(silent=True) or {}
    return _api_post("/stores/register", payload=payload)


# ---- HR data APIs ----
@app.route('/recommend', methods=['POST'])
def recommend():
    payload = request.get_json(silent=True) or {}
    return _api_post("/recommend", payload=payload)

@app.route('/rank', methods=['POST'])
def rank():
    payload = request.get_json(silent=True) or {}
    return _api_post("/rank", payload=payload)

@app.route('/list_all', methods=['GET'])
def list_all():
    params = dict(request.args.items())
    return _api_get("/list_all", params=params)


# ---- Recommendations history & save ----
@app.route('/recommendations', methods=['GET'])
def gw_recents():
    params = dict(request.args.items())
    return _api_get("/recommendations", params=params)

@app.route('/recommendations/by_store/<int:store_id>', methods=['GET'])
def gw_recents_by_store(store_id: int):
    params = dict(request.args.items())
    return _api_get(f"/recommendations/by_store/{store_id}", params=params)

@app.route('/recommendations/<int:rec_id>', methods=['GET'])
def gw_recent_detail(rec_id: int):
    return _api_get(f"/recommendations/{rec_id}")

@app.route('/recommendations/save', methods=['POST'])
def gw_save_reco():
    payload = request.get_json(silent=True) or {}
    return _api_post("/recommendations/save", payload=payload)


# ========================
# Main
# ========================
if __name__ == '__main__':
    # Gợi ý: export HR_API_URL="http://127.0.0.1:8000"
    app.run(debug=True, host='0.0.0.0', port=5000)
