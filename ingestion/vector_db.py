"""
Chroma persistence with separate collections for text (MiniLM etc.) and images (CLIP).
"""

from __future__ import annotations

from typing import List, Sequence

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import IMAGE_COLLECTION, TEXT_COLLECTION, VECTOR_DB_DIR


def _chroma_kwargs():
    return {"persist_directory": VECTOR_DB_DIR}


def ingest_text_documents(
    documents: Sequence[Document],
    embedding: HuggingFaceEmbeddings,
) -> Chroma:
    """Replace the text collection with freshly embedded documents."""
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    try:
        client.delete_collection(TEXT_COLLECTION)
    except Exception:
        pass
    return Chroma.from_documents(
        documents=list(documents),
        embedding=embedding,
        collection_name=TEXT_COLLECTION,
        **_chroma_kwargs(),
    )


def load_text_store(embedding: HuggingFaceEmbeddings) -> Chroma:
    return Chroma(
        collection_name=TEXT_COLLECTION,
        embedding_function=embedding,
        **_chroma_kwargs(),
    )


def ingest_image_documents_clip(
    documents: Sequence[Document],
    embeddings_matrix,
    ids: Sequence[str],
) -> None:
    """
    Store CLIP embeddings in a dedicated collection. `embeddings_matrix` rows must align
    with `documents` / `ids`.
    """
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    try:
        client.delete_collection(IMAGE_COLLECTION)
    except Exception:
        pass
    col = client.get_or_create_collection(name=IMAGE_COLLECTION)
    texts = [d.page_content for d in documents]
    metadatas = []
    for d in documents:
        m = {**(d.metadata or {})}
        # Chroma metadata values must be str/int/float/bool
        flat = {}
        for k, v in m.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                flat[k] = v
            else:
                flat[k] = str(v)
        metadatas.append(flat)
    col.add(
        ids=list(ids),
        embeddings=[e.tolist() for e in embeddings_matrix],
        documents=texts,
        metadatas=metadatas,
    )


def query_images_clip(query_embedding, k: int = 2) -> List[Document]:
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    try:
        col = client.get_collection(IMAGE_COLLECTION)
    except Exception:
        return []
    res = col.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs: List[Document] = []
    for text, meta in zip(res["documents"][0], res["metadatas"][0]):
        docs.append(Document(page_content=text or "", metadata=meta or {}))
    return docs
