# vector_db.py
"""
Handles creating and persisting a vector database (ChromaDB) for semantic embeddings.
"""

from langchain.vectorstores import Chroma
from config import VECTOR_DB_DIR  # Directory to store the database

def create_vector_store(chunks, embeddings):
    """
    Stores semantic embeddings in a vector database (ChromaDB).

    Args:
        chunks (list of str): List of text chunks to embed.
        embeddings: LangChain embeddings object.

    Returns:
        Chroma: Persisted vector database object.
    """
    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    db.persist()
    return db

# Example usage (for testing only)
if __name__ == "__main__":
    # from langchain.embeddings import OpenAIEmbeddings
    # embeddings = OpenAIEmbeddings()
    # chunks = ["Example text 1", "Example text 2"]
    # create_vector_store(chunks, embeddings)
    pass