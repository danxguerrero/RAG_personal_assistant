import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter

api_key = st.secrets["GOOGLE_API_KEY"]


# Upload txt files and shows a preview if loaded successfully
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode('utf-8')
    st.write("File uploaded successfully! Preview:")
    st.write(f"{text[:500]}...")

# Chunk the txt file contents using langchain
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_text(text)
st.write(f"Text successfully chunked: {chunks}")

