from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL

def get_embedder():
    """
    Returns a HuggingFace embedding model for text.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)