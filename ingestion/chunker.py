import nltk
nltk.download("punkt", quiet=True)

def semantic_chunking(documents, sentences_per_chunk=5):
    chunks = []
    for doc in documents:
        sentences = nltk.sent_tokenize(doc.page_content)
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = " ".join(sentences[i:i+sentences_per_chunk])
            if chunk.strip():
                chunks.append(chunk)
    return chunks