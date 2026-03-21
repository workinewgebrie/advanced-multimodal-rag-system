"""
Demo CLI for showing *chunking + retrieval* in the terminal (no LLM calls).

This is designed for presentations:
- It chunks a document with semantic sentence-boundary chunking.
- It embeds + indexes text in Chroma.
- It runs hybrid retrieval (dense + BM25) and prints top-k chunks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.chunker import semantic_chunk_documents
from ingestion.embedder import get_embedder
from ingestion.loader import load_document
from ingestion.vector_db import ingest_text_documents
from retrieval.hybrid import build_hybrid_retriever


def main() -> None:
    p = argparse.ArgumentParser(description="Print top retrieved chunks for a query.")
    p.add_argument("--doc", type=Path, required=True, help="Path to .pdf/.txt/.md")
    p.add_argument("--question", type=str, required=True, help="Question to retrieve context for")
    p.add_argument("--k", type=int, default=5, help="Top-k chunks to print")
    p.add_argument("--threshold", type=float, default=None, help="Semantic break threshold override")
    p.add_argument("--max-sentences", type=int, default=None, help="Max sentences per chunk override")
    p.add_argument("--dense-only", action="store_true", help="Use dense Chroma retriever only")
    args = p.parse_args()

    if not args.doc.is_file():
        raise SystemExit(f"Doc not found: {args.doc}")

    docs = load_document(args.doc)
    chunks = semantic_chunk_documents(
        docs,
        similarity_break_threshold=args.threshold
        if args.threshold is not None
        else 0.62,
        max_sentences_per_chunk=args.max_sentences
        if args.max_sentences is not None
        else 14,
    )
    print(f"[chunking] produced {len(chunks)} chunks from {args.doc.name}")
    if chunks:
        m = chunks[0].metadata or {}
        print(f"[chunking] first chunk meta: {m}")
        print(f"[chunking] first chunk preview: {chunks[0].page_content[:220].strip()}")

    embedder = get_embedder()
    vstore = ingest_text_documents(chunks, embedder)

    if args.dense_only:
        retriever = vstore.as_retriever(search_kwargs={"k": args.k})
        label = "dense-only"
    else:
        retriever = build_hybrid_retriever(vstore, chunks, k=args.k)
        label = "hybrid (dense + BM25)"

    retrieved = retriever.invoke(args.question)
    print(f"\n[retrieval] mode={label} top_k={args.k}")
    for i, d in enumerate(retrieved, 1):
        meta = d.metadata or {}
        src = meta.get("source", "?")
        page = meta.get("page")
        page_str = f" page={int(page) + 1}" if page is not None else ""
        mod = meta.get("modality", "text")
        print(f"\n[{i}] source={src}{page_str} modality={mod} chunk_index={meta.get('chunk_index')}")
        print(d.page_content[:420].strip() + ("…" if len(d.page_content) > 420 else ""))


if __name__ == "__main__":
    main()

