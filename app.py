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

# Clears the chat history
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

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

st.sidebar.header("Stored Files")

try:
    collections = chroma_db.get(include=["metadatas"])
    metadatas = collections.get("metadatas", [])

    if metadatas:
        sources = {m.get("source") for m in metadatas if m and "source" in m}
        if sources:
            for source in sources:
                st.sidebar.write(f"- {source}")
        else:
            st.sidebar.write("No files stored yet.")
    else:
        st.sidebar.write("No files stored yet.")
except Exception as e:
    st.sidebar.write(f"Error retrieving file list: {e}")


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

        metadatas = []
        for i, chunk in enumerate(chunks):
            metadatas.append({
                "source": uploaded_file.name,
                "chunk": i
            })


        chroma_db.add_texts(chunks, metadatas=metadatas)
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

    # Handle when no docs are found
    with st.expander("Debug: Retrieved Context"):
        st.write(context if context else "No context found.")

    # Build Gemini prompt
    system_instruction = "You are a helpful assistant. Use the provided context to answer the question. If you don't know the answer, say 'I don't know'."
    full_prompt = f"{system_instruction}\n\nContext:\n{context}\n\nUser Question: {prompt}"

     # Debugging: Show Gemini prompt
    with st.expander("Debug: Prompt sent to Gemini"):
        st.write(full_prompt)

    # Get Gemini response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_reply = ""

        try:
            response = model.generate_content(full_prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    full_reply += chunk.text
                    message_placeholder.markdown(full_reply + "▌")
            message_placeholder.markdown(full_reply)
        except Exception as e:
            reply = f"Error from Gemini: {e}"
            message_placeholder.markdown(reply)

    # Save + display assistant message
    st.session_state.messages.append({"role": "assistant", "content": full_reply})