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
        best: dict[str, Document] = {}
        vw = max(0.0, min(1.0, float(self.vector_weight)))
        bw = 1.0 - vw
        for rank, d in enumerate(vec_docs):
            key = _doc_key(d)
            scores[key] = scores.get(key, 0.0) + vw * (1.0 / (self.rrf_k + rank + 1))
            best.setdefault(key, d)
        for rank, d in enumerate(bm_docs):
            key = _doc_key(d)
            scores[key] = scores.get(key, 0.0) + bw * (1.0 / (self.rrf_k + rank + 1))
            best.setdefault(key, d)
        ordered = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        out: List[Document] = []
        for key in ordered[: self.k]:
            out.append(best[key])
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


# Re-export type alias for callers
HybridRetriever = RRHybridRetriever
