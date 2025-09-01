import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

api_key = st.secrets["GOOGLE_API_KEY"]
text = ""

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

if len(text) > 0:
    chunks = text_splitter.split_text(text)

# Create embeddings with Gemini
embeddings= GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=api_key
)

# Create a persisten chroma database
chroma_db = Chroma(
    collection_name="personal_docs",
    embedding_function=embeddings,
    persist_directory=".chroma_db"
)

