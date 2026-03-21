from __future__ import annotations

from langchain_core.messages import HumanMessage

from generation.provider import get_chat_llm
from generation.prompt import build_prompt


def generate_answer(context: str, question: str) -> str:
    """Single-turn helper (no chat history). Prefer `chat.rag_chain.run_rag_turn` in the app."""
    llm, _provider = get_chat_llm(temperature=0.2)
    prompt = build_prompt(context, question)
    return llm.invoke(HumanMessage(content=prompt)).content
