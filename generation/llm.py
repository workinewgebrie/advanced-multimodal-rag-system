from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from config import OPENAI_MODEL
from generation.prompt import build_prompt


def generate_answer(context: str, question: str) -> str:
    """Single-turn helper (no chat history). Prefer `chat.rag_chain.run_rag_turn` in the app."""
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
    prompt = build_prompt(context, question)
    return llm.invoke(HumanMessage(content=prompt)).content
