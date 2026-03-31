"""
Semantic chunking: split on *embedding similarity drops* between adjacent sentences,
not on fixed character counts. Small chunks are merged up to a sentence cap.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Sequence

import numpy as np
import nltk
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

# NLTK tokenizers (punkt_tab required on newer NLTK)
for _pkg in ("punkt", "punkt_tab"):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass


def _split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    try:
        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except LookupError:
        # Fallback: rough sentence boundaries if punkt data is missing
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


@lru_cache(maxsize=1)
def _chunker_model(model_name: str) -> SentenceTransformer:
    # Cached so repeated ingests in the UI don't reload the same weights.
    return SentenceTransformer(model_name)


def semantic_chunk_documents(
    documents: Sequence[Document],
    *,
    model_name: str | None = None,
    similarity_break_threshold: float = 0.62,
    min_sentences_per_chunk: int = 1,
    max_sentences_per_chunk: int = 14,
) -> List[Document]:
    """
    Break documents when consecutive sentences are *less* similar than the threshold.
    Higher threshold → more, smaller chunks. Uses the same backbone as text embeddings
    for consistent segmentation geometry.
    """
    model_name = model_name or EMBEDDING_MODEL
    model = _chunker_model(model_name)
    out: List[Document] = []
    global_idx = 0

    for doc in documents:
        sentences = _split_sentences(doc.page_content)
        if not sentences:
            continue

        embs = model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        start = 0
        for i in range(len(sentences) - 1):
            sim = _cosine_sim(embs[i], embs[i + 1])
            chunk_len = i - start + 1
            should_break = sim < similarity_break_threshold and chunk_len >= min_sentences_per_chunk
            if chunk_len >= max_sentences_per_chunk:
                should_break = True
            if should_break:
                text = " ".join(sentences[start : i + 1])
                meta = {**(doc.metadata or {}), "chunk_index": global_idx}
                out.append(Document(page_content=text, metadata=meta))
                global_idx += 1
                start = i + 1

        if start < len(sentences):
            text = " ".join(sentences[start:])
            meta = {**(doc.metadata or {}), "chunk_index": global_idx}
            out.append(Document(page_content=text, metadata=meta))
            global_idx += 1

    return out
