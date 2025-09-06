#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import requests
import pandas as pd
import streamlit as st
import time
import json

API_URL = os.getenv("HR_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="HR Recommend Chat", page_icon="🤖", layout="centered")
st.title("🤖 Chatbot Recommend Điều chuyển Nhân sự")

if "messages" not in st.session_state:
    st.session_state.messages = []
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
    st.write(f"DEBUG: Sending to /recommend: {payload}")
    r = requests.post(f"{API_URL}/recommend", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["data"]

def api_rank(top: int, intent: str = "highest_score"):
    payload = {"top": top, "intent": intent}
    st.write(f"DEBUG: Sending to /rank: {payload}")
    r = requests.post(f"{API_URL}/rank", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["data"]

def api_list_all(limit: int = 1000):
    params = {"limit": limit}
    st.write(f"DEBUG: Sending to /list_all: {params}")
    r = requests.get(f"{API_URL}/list_all", params=params, timeout=30)
    r.raise_for_status()
    return r.json()["data"]

@st.cache_data
def get_stores_df():
    try:
        data = api_stores()
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame(columns=["store_id", "store_name"])

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
    text = txt.lower()
    # Match store_id with "cho cửa hàng" or "cho store"
    m_store = re.search(r"(cho\s+(cửa hàng|store))\s*(\d+)", text)
    store_id = int(m_store.group(3)) if m_store else None

    # Match top with "top", "đề xuất", "gợi ý" before number
    m_top = re.search(r"(top|đề xuất|gợi ý)\s*(\d+)", text)
    top = int(m_top.group(2)) if m_top else 5  # Default 5

    # Match intent for ranking or list all
    intent = "recommend"  # Default intent
    if re.search(r"chỉ số thấp nhất|thấp nhất", text):
        intent = "lowest_score"
    elif re.search(r"điểm cao nhất|cao nhất", text):
        intent = "highest_score"
    elif re.search(r"show toàn bộ nhân sự|danh sách nhân sự|hiển thị tất cả", text):
        intent = "list_all"

    # Fallback store_id from last number if not a ranking or list_all intent
    if store_id is None and intent == "recommend":
        nums = re.findall(r"\d+", text)
        store_id = int(nums[-1]) if nums else None

    return store_id, top, intent

def bot_reply(msg: str):
    st.session_state.messages.append({"role": "user", "content": msg})
    store_id, top, intent = parse_intent(msg)
    st.write(f"DEBUG: Parsed store_id={store_id}, top={top}, intent={intent}")

    if store_id is None and intent == "recommend":
        store_id = st.session_state.last_store or 270
        st.session_state.messages.append({"role": "assistant", "content": f"Không tìm thấy cửa hàng cụ thể, giả sử **cửa hàng {store_id}** (bạn có thể chỉ định rõ hơn, VD: 'gợi ý cho cửa hàng 270')."})

    status_placeholder = st.empty()
    status_placeholder.write("Đang gọi API để gợi ý...")
    try:
        start_time = time.time()
        if intent == "recommend" and store_id is not None:
            data = api_recommend(store_id=store_id, top=top, exclude_same_store=True)
        elif intent in ["lowest_score", "highest_score"]:
            data = api_rank(top=top, intent=intent)
        elif intent == "list_all":
            data = api_list_all(limit=1000)  # Adjust limit as needed
        else:
            raise ValueError("Truy vấn không hợp lệ. Hãy thử 'gợi ý cho cửa hàng X', 'top N người có chỉ số thấp nhất', hoặc 'show toàn bộ nhân sự'.")

        response_time = time.time() - start_time
        st.session_state.last_store = store_id
        st.session_state.last_result = data

        df = pd.DataFrame(data)
        st.write(f"DEBUG: API returned {len(df)} rows with columns {df.columns.tolist()}")
        if not df.empty:
            if intent == "recommend":
                st.session_state.messages.append({"role": "assistant", "content": f"Top {top} ứng viên cho cửa hàng {store_id}:"})
            elif intent == "lowest_score":
                st.session_state.messages.append({"role": "assistant", "content": f"Top {top} người có chỉ số thấp nhất:"})
            elif intent == "highest_score":
                st.session_state.messages.append({"role": "assistant", "content": f"Top {top} người có điểm cao nhất:"})
            elif intent == "list_all":
                st.session_state.messages.append({"role": "assistant", "content": f"Danh sách toàn bộ nhân sự (giới hạn {len(df)} dòng):"})
            st.session_state.messages.append({"role": "assistant", "content": df})
        else:
            st.session_state.messages.append({"role": "assistant", "content": f"Không có kết quả phù hợp."})

        st.session_state.messages.append({"role": "assistant", "content": f"Thời gian phản hồi: {response_time:.2f} giây"})
        status_placeholder.empty()
        st.rerun()
    except Exception as e:
        status_placeholder.empty()
        st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Lỗi: {e}"})

# Render history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], pd.DataFrame):
            st.dataframe(message["content"], use_container_width=True)
        else:
            st.write(message["content"])

prompt = st.chat_input("Hỏi: Ví dụ 'gợi ý cho cửa hàng 270', 'top 3 cho 112', 'top 5 người có chỉ số thấp nhất', 'show toàn bộ nhân sự'")
if prompt:
    bot_reply(prompt)