from __future__ import annotations

import sys
import os
import uuid
from pathlib import Path

# Resolve imports when launched via `streamlit run app/app.py`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from chat.memory import recent_turns
from chat.rag_chain import run_rag_turn
from config import (
    DATA_IMAGES,
    DATA_RAW,
    HYBRID_VECTOR_WEIGHT,
    MAX_SENTENCES_PER_CHUNK,
    SEMANTIC_SIMILARITY_THRESHOLD,
    TOP_K,
    VECTOR_DB_DIR,
)
from ingestion.chunker import semantic_chunk_documents
from ingestion.embedder import clip_encode_image, get_embedder
from ingestion.loader import image_document, load_document
from ingestion.vector_db import ingest_image_documents_clip, ingest_text_documents
from retrieval.hybrid import build_hybrid_retriever

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

st.set_page_config(
    page_title="Multimodal RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header { font-size: 1.75rem; font-weight: 600; margin-bottom: 0.25rem; }
    .subtle { color: #666; font-size: 0.95rem; }
</style>
""",
    unsafe_allow_html=True,
)
st.markdown('<p class="main-header">Retrieval-Augmented Generation</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtle">Semantic sentence-boundary chunking · hybrid BM25+dense · optional CLIP images · history-aware chat</p>',
    unsafe_allow_html=True,
)

# --- Session defaults ---
for key, default in (
    ("messages", []),
    ("retriever", None),
    ("indexed", False),
):
    if key not in st.session_state:
        st.session_state[key] = default

# --- Sidebar: ingestion & retrieval controls ---
with st.sidebar:
    st.subheader("Ingestion")
    doc_file = st.file_uploader("Text / PDF", type=["pdf", "txt", "md"], accept_multiple_files=False)
    img_files = st.file_uploader("Images (optional, CLIP)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    img_caption_default = st.text_input("Default caption for images (if empty, filename is used)", "")

    sim_threshold = st.slider(
        "Semantic break threshold (↑ more chunks)",
        0.35,
        0.85,
        float(SEMANTIC_SIMILARITY_THRESHOLD),
        0.02,
        help="Split when cosine similarity between adjacent sentences drops below this.",
    )
    max_sent = st.slider("Max sentences per chunk", 4, 24, MAX_SENTENCES_PER_CHUNK, 1)

    st.subheader("Retrieval")
    top_k = st.number_input("Top-k (text)", min_value=1, max_value=20, value=TOP_K)
    use_hybrid = st.toggle("Hybrid search (dense + BM25)", value=True)
    vec_weight = st.slider("Dense vs BM25 (weighted RRF)", 0.0, 1.0, HYBRID_VECTOR_WEIGHT, 0.05)
    use_images = st.toggle("Retrieve related images (CLIP)", value=True)

    ingest_btn = st.button("Index uploaded files", type="primary", use_container_width=True)

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []

    st.caption(f"Vector store: `{VECTOR_DB_DIR}`")
    if not OPENAI_API_KEY:
        st.sidebar.warning("Set OPENAI_API_KEY to enable chat generation.")
    else:
        st.sidebar.caption("OpenAI key found; chat generation is enabled.")

# --- Ingestion pipeline ---
if ingest_btn:
    if not doc_file and not img_files:
        st.sidebar.error("Upload at least one document or image.")
    else:
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        DATA_IMAGES.mkdir(parents=True, exist_ok=True)

        with st.spinner("Chunking, embedding, and indexing…"):
            raw_docs = []
            if doc_file:
                p = DATA_RAW / doc_file.name
                p.write_bytes(doc_file.getbuffer())
                raw_docs.extend(load_document(p))

            chunks = semantic_chunk_documents(
                raw_docs,
                similarity_break_threshold=sim_threshold,
                max_sentences_per_chunk=max_sent,
            )

            if not chunks and img_files:
                from langchain_core.documents import Document

                chunks = [
                    Document(
                        page_content="(No text document provided; image-only index.)",
                        metadata={"source": "_placeholder"},
                    )
                ]
            elif not chunks:
                st.sidebar.error("No text could be extracted from the document.")
                st.stop()

            num_text_chunks = len(chunks)
            num_images = len(img_files) if img_files else 0

            embedder = get_embedder()
            vstore = ingest_text_documents(chunks, embedder)

            if use_hybrid:
                st.session_state.retriever = build_hybrid_retriever(
                    vstore,
                    chunks,
                    k=int(top_k),
                    vector_weight=vec_weight,
                )
            else:
                st.session_state.retriever = vstore.as_retriever(search_kwargs={"k": int(top_k)})

            if img_files:
                import numpy as np

                img_docs = []
                img_ids = []
                img_embs = []
                for f in img_files:
                    dest = DATA_IMAGES / f.name
                    dest.write_bytes(f.getbuffer())
                    cap = img_caption_default.strip() or None
                    d = image_document(dest, caption=cap)
                    img_docs.append(d)
                    img_ids.append(str(uuid.uuid4()))
                    e = clip_encode_image(str(dest))
                    img_embs.append(np.asarray(e, dtype=np.float32).reshape(-1))
                if img_embs:
                    ingest_image_documents_clip(img_docs, np.stack(img_embs, axis=0), img_ids)

            st.session_state.indexed = True
            st.session_state.last_index_summary = {
                "text_chunks": num_text_chunks,
                "images_indexed": num_images,
            }
        st.sidebar.success("Indexing complete.")
        st.sidebar.caption(
            f"Indexed: {st.session_state.last_index_summary['text_chunks']} text chunks, "
            f"{st.session_state.last_index_summary['images_indexed']} images."
        )

# --- Chat ---
if not st.session_state.indexed or st.session_state.retriever is None:
    st.info("Upload files in the sidebar and click **Index uploaded files** to start.")
    st.stop()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Ask about your materials…")
if user_q:
    if not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY. Add it to .env (or set env var) and restart the app.")
        st.stop()
    hist = recent_turns(st.session_state.messages)
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating…"):
            try:
                out = run_rag_turn(
                    user_q,
                    st.session_state.retriever,
                    hist,
                    include_images=use_images,
                )
            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.stop()

        st.markdown(out["answer"])
        st.session_state.messages.append({"role": "user", "content": user_q})
        st.session_state.messages.append({"role": "assistant", "content": out["answer"]})

        with st.expander("Sources (text)"):
            for i, d in enumerate(out.get("text_sources") or [], 1):
                meta = d.metadata or {}
                src = meta.get("source", "?")
                page = meta.get("page")
                ptxt = f", page {int(page) + 1}" if page is not None else ""
                st.markdown(f"**[{i}]** `{src}`{ptxt}")
                st.caption(d.page_content[:1200] + ("…" if len(d.page_content) > 1200 else ""))

        imgs = out.get("image_sources") or []
        if imgs:
            with st.expander("Sources (images)"):
                for d in imgs:
                    meta = d.metadata or {}
                    p = meta.get("image_path")
                    st.caption(d.page_content)
                    if p and Path(p).is_file():
                        st.image(str(p), width=320)
