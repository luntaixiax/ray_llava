import streamlit as st
import requests


def call_llava(image_uri: str = 'https://t4.ftcdn.net/jpg/07/08/47/75/360_F_708477508_DNkzRIsNFgibgCJ6KoTgJjjRZNJD4mb4.jpg',
               prompt: str = 'What is in this image?'):
    return requests.post("http://llava_service:8000/", json={"prompt": prompt, "url": image_uri}).text

st.title("LLAVA next model")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if image_uri := st.chat_input("Copy your Image link here:"):
    st.session_state.messages.append({"role": "user", "content": image_uri})
    with st.chat_message("user"):
        st.markdown(image_uri)
        st.image(image_uri) # show the image

    with st.chat_message("assistant"):
        reponse = call_llava(image_uri)
        response = st.markdown(reponse)
    st.session_state.messages.append({"role": "assistant", "content": response})