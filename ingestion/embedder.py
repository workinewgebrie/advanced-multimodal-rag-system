from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from PIL import Image
from sentence_transformers import SentenceTransformer

from config import CLIP_MODEL, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedder():
    """Dense text embeddings for the main vector index."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def _clip_model() -> SentenceTransformer:
    return SentenceTransformer(CLIP_MODEL)


def clip_encode_texts(texts: List[str]):
    model = _clip_model()
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def clip_encode_image(path: str):
    model = _clip_model()
    img = Image.open(path).convert("RGB")
    return model.encode(
        img,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
