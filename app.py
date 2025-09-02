import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

api_key = st.secrets["GOOGLE_API_KEY"]


st.title("Personal RAG Assitant")
st.write("Upload a '.txt' or a '.pdf' file and I'll remember them for you.")
text = ""

# Upload txt and pdf files and shows a preview if loaded successfully
uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

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


if uploaded_file:
    st.write(f"Processing: {uploaded_file.name}")


    # Handle PDF uploads
    if uploaded_file.type == "application/pdf":
        pdf = PdfReader(uploaded_file)
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted

    # Handl TXT uploads
    elif uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode('utf-8')
    
    st.write("File uploaded successfully! Preview:")
    st.write(f"{text[:500]}...")

    # Chunk the file's text content using langchain
    if text.strip():
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = text_splitter.split_text(text)


        chroma_db.add_texts(chunks)
        st.success("Chunked and stored in Chroma!")
    else:
        st.warning("No text could be extracted from this file.")
    
