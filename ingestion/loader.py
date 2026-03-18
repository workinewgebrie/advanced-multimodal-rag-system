from langchain.document_loaders import PyPDFLoader, TextLoader

def load_document(path: str):
    if path.endswith(".pdf"):
        return PyPDFLoader(path).load()
    elif path.endswith(".txt"):
        return TextLoader(path).load()
    else:
        raise ValueError("Unsupported file type")