from __future__ import annotations

import sys
import uuid
from datetime import datetime
from pathlib import Path

# Resolve imports when launched via `streamlit run app/app.py`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

load_dotenv(_ROOT / ".env")

from chat.memory import recent_turns
from chat.rag_chain import retrieve_context, run_rag_turn
from config import (
    DATA_IMAGES,
    DATA_RAW,
    GEMINI_API_KEY,
    HYBRID_VECTOR_WEIGHT,
    MAX_SENTENCES_PER_CHUNK,
    OPENAI_API_KEY,
    SEMANTIC_SIMILARITY_THRESHOLD,
    TOP_K,
    VECTOR_DB_DIR,
)
from ingestion.chunker import semantic_chunk_documents
from ingestion.embedder import clip_encode_image, get_embedder
from ingestion.loader import image_document, load_document
from ingestion.vector_db import ingest_image_documents_clip, ingest_text_documents
from retrieval.hybrid import build_hybrid_retriever

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
    ("last_chunks_preview", []),
    ("last_images_preview", []),
    ("last_index_summary", {}),
    ("last_uploaded_image_paths", []),
    ("last_generated_image_path", ""),
):
    if key not in st.session_state:
        st.session_state[key] = default


def _generate_similar_demo_image(source_path: Path, output_dir: Path) -> Path:
    """
    Create a stylistic variation of an uploaded image for demo purposes.
    This is local generation (no external API), intended to showcase
    image-generation-like UX in presentations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(source_path).convert("RGB")

    # Keep original size but apply deterministic visual variation.
    base = ImageOps.autocontrast(img)
    base = ImageEnhance.Color(base).enhance(1.25)
    base = ImageEnhance.Contrast(base).enhance(1.15)
    base = base.filter(ImageFilter.DETAIL)

    # Blend with a mirrored soft layer for a "similar but new" look.
    mirror = ImageOps.mirror(base).filter(ImageFilter.GaussianBlur(radius=2))
    out = Image.blend(base, mirror, alpha=0.22)

    # Add a subtle frame for demo visibility.
    framed = Image.new("RGB", (out.width + 24, out.height + 24), (238, 240, 245))
    framed.paste(out, (12, 12))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = output_dir / f"generated_similar_{ts}.png"
    framed.save(dest)
    return dest

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
    if not GEMINI_API_KEY and not OPENAI_API_KEY:
        st.sidebar.warning("Set GEMINI_API_KEY or OPENAI_API_KEY to enable chat generation.")
    else:
        provider = "Gemini" if GEMINI_API_KEY else "OpenAI"
        st.sidebar.caption(f"{provider} key found; chat generation is enabled.")

    if st.session_state.get("messages"):
        chat_text = "\n".join(
            [f"{m['role'].upper()}: {m.get('content','')}" for m in st.session_state.messages]
        )
        st.download_button(
            "Download chat transcript",
            chat_text,
            file_name="chat_history.txt",
            mime="text/plain",
            use_container_width=True,
        )

    if st.button("Generate similar demo image", use_container_width=True):
        paths = st.session_state.get("last_uploaded_image_paths") or []
        if not paths:
            st.sidebar.warning("Upload and index at least one image first.")
        else:
            src = Path(paths[0])
            try:
                out_img = _generate_similar_demo_image(src, DATA_IMAGES)
                st.session_state.last_generated_image_path = str(out_img)
                st.sidebar.success("Generated a similar demo image.")
            except Exception as e:
                st.sidebar.error(f"Image generation failed: {e}")

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

            # Keep small previews for the instructor demo.
            st.session_state.last_chunks_preview = [
                {
                    "chunk_index": d.metadata.get("chunk_index") if d.metadata else None,
                    "source": (d.metadata or {}).get("source", "?"),
                    "text": d.page_content[:180] + ("…" if len(d.page_content) > 180 else ""),
                }
                for d in chunks[:6]
            ]
            st.session_state.last_images_preview = []

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
                uploaded_paths = []
                for f in img_files:
                    dest = DATA_IMAGES / f.name
                    dest.write_bytes(f.getbuffer())
                    uploaded_paths.append(str(dest.resolve()))
                    cap = img_caption_default.strip() or None
                    d = image_document(dest, caption=cap)
                    img_docs.append(d)
                    img_ids.append(str(uuid.uuid4()))
                    e = clip_encode_image(str(dest))
                    img_embs.append(np.asarray(e, dtype=np.float32).reshape(-1))
                if img_embs:
                    ingest_image_documents_clip(img_docs, np.stack(img_embs, axis=0), img_ids)
                st.session_state.last_images_preview = [
                    {
                        "caption": d.page_content,
                        "image_path": (d.metadata or {}).get("image_path"),
                    }
                    for d in img_docs[:6]
                ]
                st.session_state.last_uploaded_image_paths = uploaded_paths

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

        if st.session_state.last_chunks_preview:
            with st.expander("Chunking preview (first 6 chunks)", expanded=False):
                for c in st.session_state.last_chunks_preview:
                    st.markdown(
                        f"- chunk `{c.get('chunk_index')}` from `{c.get('source')}`"
                    )
                    st.caption(c.get("text", ""))

        if st.session_state.last_images_preview:
            with st.expander("Image preview (first 6 images)", expanded=False):
                for im in st.session_state.last_images_preview:
                    st.caption(im.get("caption", ""))
                    img_path = im.get("image_path")
                    if img_path and Path(img_path).is_file():
                        st.image(str(img_path), width=160)

        generated_path = st.session_state.get("last_generated_image_path")
        if generated_path and Path(generated_path).is_file():
            with st.expander("Generated similar image", expanded=True):
                st.caption("Local synthetic generation from first uploaded image.")
                st.image(generated_path, width=320)

# --- Chat ---
if not st.session_state.indexed or st.session_state.retriever is None:
    st.info("Upload files in the sidebar and click **Index uploaded files** to start.")
    st.stop()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Ask about your materials…")
if user_q:
    if not GEMINI_API_KEY and not OPENAI_API_KEY:
        st.error("Missing GEMINI_API_KEY or OPENAI_API_KEY. Add it to .env (or set env var) and restart the app.")
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
                # Graceful fallback for quota/key/provider issues: still show retrieval output.
                text_docs, image_docs = retrieve_context(
                    user_q,
                    st.session_state.retriever,
                    include_images=use_images,
                )
                preview_lines = []
                for i, d in enumerate(text_docs[:3], 1):
                    snippet = d.page_content.strip().replace("\n", " ")
                    if len(snippet) > 220:
                        snippet = snippet[:220] + "…"
                    preview_lines.append(f"{i}. {snippet}")
                fallback_answer = (
                    "Generation is currently unavailable (likely API quota/key/provider issue). "
                    "Showing retrieval-only result so you can continue the demo.\n\n"
                    + ("\n".join(preview_lines) if preview_lines else "No retrieved text chunks.")
                )
                out = {
                    "answer": fallback_answer,
                    "text_sources": text_docs,
                    "image_sources": image_docs,
                    "timing": {},
                    "provider": "retrieval-only-fallback",
                }
                st.warning(f"Generation failed, fallback active: {e}")

        st.markdown(out["answer"])
        timing = out.get("timing") or {}
        retrieval_s = timing.get("retrieval_s")
        generation_s = timing.get("generation_s")
        provider = out.get("provider")
        if retrieval_s is not None and generation_s is not None:
            provider_txt = f" · provider {provider}" if provider else ""
            st.caption(
                f"Timing: retrieval {retrieval_s:.2f}s · generation {generation_s:.2f}s{provider_txt}"
            )
        st.session_state.messages.append({"role": "user", "content": user_q})
        st.session_state.messages.append({"role": "assistant", "content": out["answer"]})

        with st.expander("Sources (text)"):
            for i, d in enumerate(out.get("text_sources") or [], 1):
                meta = d.metadata or {}
                src = meta.get("source", "?")
                page = meta.get("page")
                ptxt = f", page {int(page) + 1}" if page is not None else ""
                dense_rank = meta.get("dense_rank")
                bm25_rank = meta.get("bm25_rank")
                origin_bits = []
                if dense_rank is not None:
                    origin_bits.append(f"dense#{int(dense_rank)+1}")
                if bm25_rank is not None:
                    origin_bits.append(f"bm25#{int(bm25_rank)+1}")
                origin = "+".join(origin_bits) if origin_bits else "vector/bm25"
                origin_txt = f" · {origin}"
                st.markdown(f"**[{i}]** `{src}`{ptxt}{origin_txt}")
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
