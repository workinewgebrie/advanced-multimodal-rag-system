from langchain_community.document_loaders import PyPDFLoader, TextLoader
from typing import List

def load_document(path: str):
    """
    Load PDF or TXT documents using LangChain community loaders.
    """
    if path.endswith(".pdf"):
        loader = PyPDFLoader(path)
        return loader.load()

    elif path.endswith(".txt"):
        loader = TextLoader(path, encoding="utf-8")
        return loader.load()

    else:
        raise ValueError("Unsupported file type. Use PDF or TXT.")