def build_prompt(context: str, question: str) -> str:
    return f"""
Use ONLY the context below to answer the question.

Context:
{context}

Question:
{question}

Answer clearly and accurately.
"""