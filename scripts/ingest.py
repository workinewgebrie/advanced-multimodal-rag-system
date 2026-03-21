"""
CLI: ingest a PDF or text file into the Chroma text collection (same paths as the UI).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.chunker import semantic_chunk_documents
from ingestion.embedder import get_embedder
from ingestion.loader import load_document
from ingestion.vector_db import ingest_text_documents
from retrieval.hybrid import build_hybrid_retriever


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest a document into Chroma (text index).")
    p.add_argument("path", type=Path, help="Path to .pdf, .txt, or .md")
    p.add_argument("--threshold", type=float, default=None, help="Semantic similarity break threshold")
    p.add_argument("--max-sentences", type=int, default=None, help="Max sentences per chunk")
    args = p.parse_args()
    path: Path = args.path
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")

    docs = load_document(path)
    kwargs = {}
    if args.threshold is not None:
        kwargs["similarity_break_threshold"] = args.threshold
    if args.max_sentences is not None:
        kwargs["max_sentences_per_chunk"] = args.max_sentences
    chunks = semantic_chunk_documents(docs, **kwargs)
    embedder = get_embedder()
    vstore = ingest_text_documents(chunks, embedder)
    _ = build_hybrid_retriever(vstore, chunks)
    print(f"Ingested {len(chunks)} chunks from {path.name} into vector store.")


if __name__ == "__main__":
    main()
