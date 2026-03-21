"""
Hybrid retrieval: dense Chroma + lexical BM25 merged via Reciprocal Rank Fusion (RRF).
Avoids deprecated / relocated EnsembleRetriever APIs across LangChain versions.
"""

from __future__ import annotations

from typing import List, Sequence

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import TOP_K


def _doc_key(d: Document) -> str:
    meta = d.metadata or {}
    return f"{d.page_content}\n{meta.get('source')}|{meta.get('page')}|{meta.get('chunk_index')}"


class RRHybridRetriever(BaseRetriever):
    """Weighted RRF merge of dense vector search and BM25 over the same chunked corpus."""

    vector_retriever: BaseRetriever
    bm25_retriever: BM25Retriever
    k: int = TOP_K
    rrf_k: int = 60
    vector_weight: float = 0.55

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        vec_docs = self.vector_retriever.invoke(query)
        bm_docs = self.bm25_retriever.invoke(query)
        scores: dict[str, float] = {}
        best_docs: dict[str, Document] = {}
        best_meta: dict[str, dict] = {}
        vw = max(0.0, min(1.0, float(self.vector_weight)))
        bw = 1.0 - vw
        for rank, d in enumerate(vec_docs):
            key = _doc_key(d)
            scores[key] = scores.get(key, 0.0) + vw * (1.0 / (self.rrf_k + rank + 1))
            best_docs.setdefault(key, d)
            meta = best_meta.setdefault(key, dict(d.metadata or {}))
            # Track best (lowest) dense rank for explainability in UI.
            dense_rank = rank
            prev = meta.get("dense_rank")
            meta["dense_rank"] = dense_rank if prev is None else min(int(prev), dense_rank)
        for rank, d in enumerate(bm_docs):
            key = _doc_key(d)
            scores[key] = scores.get(key, 0.0) + bw * (1.0 / (self.rrf_k + rank + 1))
            best_docs.setdefault(key, d)
            meta = best_meta.setdefault(key, dict(d.metadata or {}))
            bm25_rank = rank
            prev = meta.get("bm25_rank")
            meta["bm25_rank"] = bm25_rank if prev is None else min(int(prev), bm25_rank)
        ordered = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        out: List[Document] = []
        for key in ordered[: self.k]:
            base = best_docs[key]
            combined_meta = dict(base.metadata or {})
            combined_meta.update(best_meta.get(key, {}))
            out.append(Document(page_content=base.page_content, metadata=combined_meta))
        return out


def build_hybrid_retriever(
    vector_store: Chroma,
    chunked_documents: Sequence[Document],
    *,
    k: int | None = None,
    vector_weight: float | None = None,
) -> RRHybridRetriever:
    """Larger `vector_weight` emphasizes dense semantic matches vs lexical BM25."""
    from config import HYBRID_VECTOR_WEIGHT

    k = k if k is not None else TOP_K
    vw = vector_weight if vector_weight is not None else HYBRID_VECTOR_WEIGHT
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    bm25 = BM25Retriever.from_documents(list(chunked_documents))
    bm25.k = k
    return RRHybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25,
        k=k,
        vector_weight=vw,
    )
