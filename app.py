import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import google.generativeai as genai

api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")


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

# If we don't have messages for our session, initialize an empty list
if "messages" not in st.session_state:
    st.session_state.messages = []

# Loop through each message and display them in the chat
for msg in st.session_state.messages:
    role = msg["role"]
    st.chat_message(role).write(msg["content"])

if prompt := st.chat_input("Ask me something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Retrieve relevant docs
    docs = chroma_db.similarity_search(prompt, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # Build Gemini prompt
    system_instruction = "You are a helpful assistant. Use the provided context to answer the question."
    full_prompt = f"{system_instruction}\n\nContext:\n{context}\n\nUser Question: {prompt}"

    # Get Gemini response
    response = model.generate_content(full_prompt)
    reply = response.text

    # Save + display assistant message
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assitant").write(reply)