from __future__ import annotations

from typing import List, Sequence, Tuple

from langchain_core.documents import Document


def _format_sources(docs: Sequence[Document]) -> Tuple[str, List[str]]:
    """Build a numbered context block and parallel citation keys."""
    lines: List[str] = []
    keys: List[str] = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        page = meta.get("page")
        mod = meta.get("modality", "text")
        head = f"[{i}] source={src}"
        if page is not None:
            head += f" page={int(page) + 1}"  # PDF pages are 0-based in PyPDFLoader
        if mod:
            head += f" modality={mod}"
        lines.append(f"{head}\n{d.page_content}")
        keys.append(f"[{i}]")
    return "\n\n---\n\n".join(lines), keys


def build_messages(
    *,
    question: str,
    context_docs: Sequence[Document],
    chat_history: List[dict],
) -> list:
    """
    Chat messages for OpenAI-compatible chat models: system rubric + optional history
    + user message with grounded context.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    context_block, _keys = _format_sources(context_docs)
    system = SystemMessage(
        content=(
            "You are a careful assistant for document Q&A. "
            "Answer using ONLY the provided context and conversation so far. "
            "If the answer is not contained in the context, say you don't have enough "
            "information in the indexed materials — do not invent facts. "
            "Cite sources inline using the bracket labels like [1], [2] where appropriate."
        )
    )
    msgs = [system]
    for turn in chat_history:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    user = HumanMessage(
        content=(
            f"Context from retrieval:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            "Answer concisely and cite relevant source indices."
        )
    )
    msgs.append(user)
    return msgs


# Backwards-compatible helper for simple non-chat calls
def build_prompt(context: str, question: str) -> str:
    return f"""
Use ONLY the context below to answer the question.

Context:
{context}

Question:
{question}

Answer clearly and accurately.
"""
