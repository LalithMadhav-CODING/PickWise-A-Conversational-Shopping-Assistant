"""The model. Adapted from Module 4 · Agentic_app (relay_ops/llm.py).

Probed on qwen3:4b (see CURRENT_PROJECT_STATUS.md, Phase 0):
- reasoning=True  -> clean `content`; the thinking trace is in additional_kwargs["reasoning_content"]
- reasoning=False -> Qwen3 still thinks, but the "Okay, let's see..." narration leaks into `content`
"""
from functools import lru_cache

from langchain_ollama import ChatOllama

from . import config


@lru_cache(maxsize=8)
def get_llm(model: str = config.MODEL, reasoning: bool = config.REASONING) -> ChatOllama:
    # temperature=0 -> repeatable runs
    return ChatOllama(model=model, base_url=config.OLLAMA_URL, temperature=0, reasoning=reasoning)


def reasoning_trace(message) -> str:
    """The model's thinking for one reply ('' when reasoning is off)."""
    return message.additional_kwargs.get("reasoning_content", "")
