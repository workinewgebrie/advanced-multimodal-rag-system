from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config import OPENAI_MODEL, TOP_K, TOP_K_IMAGES
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
    text_docs, image_docs = retrieve_context(
        question,
        retriever,
        include_images=include_images,
    )
    context_docs: List[Document] = list(text_docs) + list(image_docs)

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
    messages = build_messages(
        question=question,
        context_docs=context_docs,
        chat_history=chat_history,
    )
    resp = llm.invoke(messages)
    return {
        "answer": resp.content,
        "text_sources": text_docs,
        "image_sources": image_docs,
    }


def build_rag_chain(retriever):
    """
    Legacy hook kept for imports; Streamlit uses run_rag_turn directly.
    """
    return retriever
