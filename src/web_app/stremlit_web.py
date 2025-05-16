import streamlit as st
import requests

st.set_page_config(page_title="CHATBOT LUẬT", layout="centered")
st.title("💬 CHATBOT HỎI ĐÁP LUẬT")

# Lưu lịch sử hội thoại
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị đoạn chat trước (ở trên)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Nhận input từ người dùng (textbox nằm dưới)
user_input = st.chat_input("Nhập câu hỏi của bạn...")

if user_input:
    # Hiển thị tin nhắn người dùng
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Gửi đến API Flask
    try:
        res = requests.post(
            "http://localhost:5001/api/chatbot",
            json={"query": user_input},
            timeout=10
        )
        res.raise_for_status()
        data = res.json()
        bot_reply = data.get("answer", str(data))
    except Exception as e:
        bot_reply = f"Lỗi: {e}"

    # Hiển thị phản hồi từ bot
    st.chat_message("assistant").markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
