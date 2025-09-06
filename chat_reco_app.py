#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import requests
import pandas as pd
import streamlit as st

API_URL = os.getenv("HR_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="HR Recommend Chat", page_icon="🤖", layout="centered")
st.title("🤖 Chatbot Recommend Điều chuyển Nhân sự")

if "history" not in st.session_state:
    st.session_state.history = []
if "last_store" not in st.session_state:
    st.session_state.last_store = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

def api_stores():
    r = requests.get(f"{API_URL}/stores", timeout=10)
    r.raise_for_status()
    return r.json()

def api_recommend(store_id: int, top: int = 5, exclude_same_store: bool = True):
    payload = {"store_id": store_id, "top": top, "exclude_same_store": exclude_same_store}
    r = requests.post(f"{API_URL}/recommend", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["data"]

@st.cache_data
def get_stores_df():
    try:
        data = api_stores()
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame(columns=["store_id","store_name"])

stores_df = get_stores_df()

with st.sidebar:
    st.subheader("Tùy chọn")
    st.write("API:", API_URL)
    if not stores_df.empty:
        st.dataframe(stores_df, use_container_width=True, height=200)
    st.markdown("---")
    if st.session_state.last_result is not None:
        csv = pd.DataFrame(st.session_state.last_result).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Tải CSV kết quả gần nhất", data=csv, file_name="recommendations.csv", mime="text/csv")

def parse_intent(txt: str):
    """
    Trích store_id và top N từ câu tiếng Việt đơn giản:
    - 'gợi ý cho cửa hàng 270', 'top 3 cho 112', 'đề xuất 5 cho store 24', ...
    """
    text = txt.lower()
    # top N
    m_top = re.search(r"(top|top\s*)\s*(\d+)", text)
    top = int(m_top.group(2)) if m_top else None

    # store_id (ưu tiên 'cửa hàng <id>' / 'store <id>')
    m_store = re.search(r"(cửa hàng|store)\s*(\d+)", text)
    if m_store:
        store_id = int(m_store.group(2))
    else:
        # fallback: số cuối câu
        nums = re.findall(r"\d+", text)
        store_id = int(nums[-1]) if nums else None

    return store_id, top

def bot_reply(msg: str):
    st.session_state.history.append(("user", msg))

    store_id, top = parse_intent(msg)
    if top is None:
        top = 5

    if store_id is None:
        if st.session_state.last_store is not None:
            store_id = st.session_state.last_store
        else:
            st.session_state.history.append(("bot", "Bạn muốn gợi ý cho **cửa hàng nào** (VD: *cửa hàng 270*, *top 3 cho 112*)?"))
            return

    try:
        data = api_recommend(store_id=store_id, top=top, exclude_same_store=True)
        st.session_state.last_store = store_id
        st.session_state.last_result = data

        df = pd.DataFrame(data)
        if not df.empty:
            st.session_state.history.append(("bot", f"Top {top} ứng viên cho **cửa hàng {store_id}**:"))
            st.session_state.history.append(("table", df))
        else:
            st.session_state.history.append(("bot", f"Không có ứng viên phù hợp cho cửa hàng {store_id}."))
    except Exception as e:
        st.session_state.history.append(("bot", f"⚠️ Lỗi: {e}"))

# Chat box
for role, payload in st.session_state.history:
    if role == "user":
        with st.chat_message("user"): st.write(payload)
    elif role == "bot":
        with st.chat_message("assistant"): st.write(payload)
    elif role == "table":
        st.dataframe(payload, use_container_width=True)

prompt = st.chat_input("Hỏi: Ví dụ 'gợi ý cho cửa hàng 270', 'top 3 cho 112'")
if prompt:
    bot_reply(prompt)
