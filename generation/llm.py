from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from config import GEMINI_API_KEY, GEMINI_MODEL, OPENAI_API_KEY, OPENAI_MODEL
from generation.prompt import build_prompt


def generate_answer(context: str, question: str) -> str:
    """Single-turn helper (no chat history). Prefer `chat.rag_chain.run_rag_turn` in the app."""
    if GEMINI_API_KEY:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            api_key=GEMINI_API_KEY,
            temperature=0.2,
        )
    elif OPENAI_API_KEY:
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
    else:
        raise ValueError("Missing GEMINI_API_KEY and OPENAI_API_KEY; cannot generate answers.")
    prompt = build_prompt(context, question)
    return llm.invoke(HumanMessage(content=prompt)).content
