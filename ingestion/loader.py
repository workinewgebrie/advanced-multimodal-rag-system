from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def load_document(path: str | Path) -> List[Document]:
    """
    Load PDF (per-page metadata) or UTF-8 text. Paths must exist on disk.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        for d in docs:
            d.metadata = {**(d.metadata or {}), "source": path.name}
        return docs

    if suffix in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata = {**(d.metadata or {}), "source": path.name}
        return docs

    raise ValueError("Unsupported file type. Use PDF, TXT, or MD.")


def image_document(image_path: str | Path, caption: str | None = None) -> Document:
    """
    Build a LangChain Document for an image. The page_content is what the LLM will see;
    CLIP embeddings use the same string for the text tower when no pixels are encoded
    in the text index (images go to the image collection).
    """
    path = Path(image_path)
    label = caption.strip() if caption and caption.strip() else f"Image file: {path.name}"
    return Document(
        page_content=label,
        metadata={
            "source": path.name,
            "modality": "image",
            "image_path": str(path.resolve()),
        },
    )
