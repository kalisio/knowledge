"""LLM client abstraction used by the API service."""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests

from src.rag_system.config import ApiConfig


SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant for the Kalisio codebase.\n"
    "Answer the user's question using only the provided context.\n"
    "Do not use outside knowledge.\n"
    "If the context does not contain enough information, say: "
    "\"I do not know based on the provided Kalisio context.\"\n"
    "Keep the answer concise and mention relevant source paths when useful."
)


@dataclass(frozen=True)
class LLMResponse:
    answer: str
    provider: str
    model: str


class LLMClient:
    def __init__(self, config: ApiConfig):
        self.config = config

    def ask(self, prompt: str) -> LLMResponse:
        provider = self.config.llm_provider.lower()
        if provider == "anthropic":
            return self._ask_anthropic(prompt)
        return self._ask_ollama(prompt)

    def _ask_ollama(self, prompt: str) -> LLMResponse:
        response = requests.post(
            f"{self.config.ollama_base_url.rstrip('/')}/api/generate",
            json={
                "model": self.config.ollama_model,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {"num_predict": self.config.max_answer_tokens},
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        return LLMResponse(
            answer=str(payload.get("response", "")).strip(),
            provider="ollama",
            model=self.config.ollama_model,
        )

    def _ask_anthropic(self, prompt: str) -> LLMResponse:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("Install anthropic to use LLM_PROVIDER=anthropic") from exc

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=self.config.anthropic_model,
            max_tokens=self.config.max_answer_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = "\n".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        ).strip()
        return LLMResponse(
            answer=answer,
            provider="anthropic",
            model=self.config.anthropic_model,
        )

