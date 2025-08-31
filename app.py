import streamlit as st

api_key = st.secrets["GOOGLE_API_KEY"]

uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode('utf-8')
    st.write("File uploaded successfully! Preview:")
    st.write(f"{text[:500]}...")

    