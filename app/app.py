import streamlit as st
import os

from ingestion.loader import load_document
from ingestion.chunker import semantic_chunking
from ingestion.embedder import get_embedder
from ingestion.vector_db import create_vector_store
from chat.rag_chain import build_rag_chain
from config import TOP_K

st.set_page_config(
    page_title="Advanced Multimodal RAG System",
    layout="wide"
)

st.title("🧠 Advanced Multimodal RAG System")
st.markdown("A history-aware Retrieval-Augmented Generation system")

# -----------------------
# FILE UPLOAD
# -----------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload a document (PDF or TXT)",
    type=["pdf", "txt"]
)

if uploaded_file:
    os.makedirs("data/raw", exist_ok=True)
    file_path = f"data/raw/{uploaded_file.name}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Ingesting document..."):
        documents = load_document(file_path)
        chunks = semantic_chunking(documents)
        embedder = get_embedder()

        db = create_vector_store(chunks, embedder)
        retriever = db.as_retriever(search_kwargs={"k": TOP_K})

        st.session_state.chain = build_rag_chain(retriever)

    st.success("Document successfully indexed!")

# -----------------------
# CHAT INTERFACE
# -----------------------
if "chain" in st.session_state:
    query = st.text_input("Ask a question about the document")

    if query:
        with st.spinner("Generating answer..."):
            response = st.session_state.chain({"question": query})

        st.subheader("Answer")
        st.write(response["answer"])

        with st.expander("Sources"):
            for doc in response["source_documents"]:
                st.write(doc.page_content[:300])