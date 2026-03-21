from __future__ import annotations

import time
from typing import List, Sequence, Tuple

import numpy as np
from langchain_core.documents import Document

from config import TOP_K, TOP_K_IMAGES
from generation.provider import get_chat_llm
from generation.prompt import build_messages
from ingestion.embedder import clip_encode_texts
from ingestion.vector_db import query_images_clip


def _dedupe_docs(docs: Sequence[Document], limit: int) -> List[Document]:
    seen = set()
    out: List[Document] = []
    for d in docs:
        key = (d.page_content, str(d.metadata))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
        if len(out) >= limit:
            break
    return out


def retrieve_context(
    question: str,
    retriever,
    *,
    k_text: int | None = None,
    include_images: bool = True,
) -> Tuple[List[Document], List[Document]]:
    """
    Returns (text_docs, image_docs). Image docs come from CLIP similarity when enabled.
    """
    k_text = k_text or TOP_K
    text_docs = retriever.invoke(question)
    text_docs = _dedupe_docs(text_docs, k_text)

    image_docs: List[Document] = []
    if include_images:
        q_emb = clip_encode_texts([question])[0]
        # Ensure 1-D numpy
        qv = np.asarray(q_emb, dtype=np.float32).reshape(-1)
        image_docs = query_images_clip(qv, k=TOP_K_IMAGES)
    return text_docs, image_docs


def run_rag_turn(
    question: str,
    retriever,
    chat_history: List[dict],
    *,
    include_images: bool = True,
) -> dict:
    """
    History-aware RAG turn. `chat_history` is prior turns only (exclude current question).
    Returns dict with answer, text_sources, image_sources.
    """
    retrieval_start = time.perf_counter()
    text_docs, image_docs = retrieve_context(
        question,
        retriever,
        include_images=include_images,
    )
    retrieval_s = time.perf_counter() - retrieval_start
    context_docs: List[Document] = list(text_docs) + list(image_docs)

    generation_start = time.perf_counter()
    llm, provider = get_chat_llm(temperature=0.2)
    messages = build_messages(
        question=question,
        context_docs=context_docs,
        chat_history=chat_history,
    )
    resp = llm.invoke(messages)
    generation_s = time.perf_counter() - generation_start
    return {
        "answer": resp.content,
        "text_sources": text_docs,
        "image_sources": image_docs,
        "timing": {
            "retrieval_s": retrieval_s,
            "generation_s": generation_s,
            "total_s": retrieval_s + generation_s,
        },
        "provider": provider,
    }


def build_rag_chain(retriever):
    """
    Legacy hook kept for imports; Streamlit uses run_rag_turn directly.
    """
    return retriever
