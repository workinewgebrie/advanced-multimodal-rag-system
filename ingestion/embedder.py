from langchain.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL

def get_text_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)