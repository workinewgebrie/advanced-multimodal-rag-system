from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from config import GEMINI_API_KEY, GEMINI_MODEL, OPENAI_API_KEY, OPENAI_MODEL


def _norm_gemini_name(name: str) -> str:
    return (name or "").replace("models/", "").strip()


def _candidate_models() -> list[str]:
    preferred = [
        _norm_gemini_name(GEMINI_MODEL),
        "gemini-2.0-flash",
        "gemini-flash-latest",
        "gemini-2.5-flash",
    ]
    out = []
    seen = set()
    for m in preferred:
        if m and m not in seen:
            out.append(m)
            seen.add(m)
    return out


@lru_cache(maxsize=1)
def _available_gemini_models() -> set[str]:
    """
    Ask Gemini which models are available for this key.
    If listing fails, return empty and let caller fall back to configured model.
    """
    if not GEMINI_API_KEY:
        return set()
    try:
        from google import genai

        client = genai.Client(api_key=GEMINI_API_KEY)
        out = set()
        for m in client.models.list():
            out.add(_norm_gemini_name(getattr(m, "name", "")))
        return out
    except Exception:
        return set()


def _first_available(candidates: Iterable[str], available: set[str]) -> str:
    for c in candidates:
        if c in available:
            return c
    # If list API failed or none matched, use first candidate and let runtime error be explicit.
    for c in candidates:
        return c
    return "gemini-2.0-flash"


def get_chat_llm(temperature: float = 0.2) -> Tuple[object, str]:
    """
    Returns (llm, provider_label). Prefers Gemini, falls back to OpenAI.
    """
    if GEMINI_API_KEY:
        model = _first_available(_candidate_models(), _available_gemini_models())
        return (
            ChatGoogleGenerativeAI(
                model=model,
                api_key=GEMINI_API_KEY,
                temperature=temperature,
            ),
            f"gemini:{model}",
        )
    if OPENAI_API_KEY:
        return ChatOpenAI(model=OPENAI_MODEL, temperature=temperature), f"openai:{OPENAI_MODEL}"
    raise ValueError("Missing GEMINI_API_KEY and OPENAI_API_KEY; cannot generate answers.")

