from __future__ import annotations

import os
from typing import Iterable

from langchain_core.messages import BaseMessage
from langchain_mistralai import ChatMistralAI


DEFAULT_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "mistral-large-latest")
DEFAULT_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "6"))


class LLMClient:
    """Wrapper that mirrors the notebook's LangChain-based LLM interface."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        api_key: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Set MISTRAL_API_KEY or provide api_key to LLMClient in order to call the LLM."
            )
        self.model = ChatMistralAI(
            model=model_name,
            api_key=self.api_key,
            max_retries=max_retries,
        )

    def generate(self, messages: Iterable[BaseMessage]) -> str:
        response = None
        while response is None:
            try:
                response = self.model.invoke(list(messages))
            except Exception as exc:
                if "service_tier_capacity_exceeded" in str(exc):
                    continue
                raise
        return response.content
