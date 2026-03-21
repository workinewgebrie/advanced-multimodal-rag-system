"""
Hybrid retrieval: dense vector search + lexical BM25, merged with weighted ranks.
"""

from __future__ import annotations

from typing import List, Sequence

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import HYBRID_VECTOR_WEIGHT, TOP_K


def build_hybrid_retriever(
    vector_store: Chroma,
    chunked_documents: Sequence[Document],
    *,
    k: int | None = None,
    vector_weight: float | None = None,
) -> EnsembleRetriever:
    k = k if k is not None else TOP_K
    w = vector_weight if vector_weight is not None else HYBRID_VECTOR_WEIGHT
    w = max(0.0, min(1.0, w))
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    bm25 = BM25Retriever.from_documents(list(chunked_documents))
    bm25.k = k
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25],
        weights=[w, 1.0 - w],
    )
