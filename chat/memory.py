from __future__ import annotations

from typing import List

from config import MAX_HISTORY_TURNS


def recent_turns(messages: List[dict], max_pairs: int | None = None) -> List[dict]:
    """
    Keep the last N user/assistant pairs for the LLM. `messages` uses OpenAI-style
    dicts: {"role": "user"|"assistant", "content": str}.
    """
    max_pairs = max_pairs if max_pairs is not None else MAX_HISTORY_TURNS
    cap = max(0, max_pairs) * 2
    if cap == 0:
        return []
    return messages[-cap:] if len(messages) > cap else list(messages)
