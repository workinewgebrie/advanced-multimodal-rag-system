import os
from pathlib import Path

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CLIP_MODEL = os.getenv("CLIP_MODEL", "clip-ViT-B-32")

# Vector store (single persist root; separate Chroma collections for text vs images)
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", str(Path("vectorstore") / "chroma"))
TEXT_COLLECTION = "rag_text"
IMAGE_COLLECTION = "rag_images"

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
TOP_K_IMAGES = int(os.getenv("TOP_K_IMAGES", "2"))
HYBRID_VECTOR_WEIGHT = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.55"))

# Chunking (similarity on adjacent sentences; higher → more splits)
SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.62"))
MAX_SENTENCES_PER_CHUNK = int(os.getenv("MAX_SENTENCES_PER_CHUNK", "14"))

# Generation
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "8"))

DATA_RAW = Path("data") / "raw"
DATA_IMAGES = Path("data") / "images"
