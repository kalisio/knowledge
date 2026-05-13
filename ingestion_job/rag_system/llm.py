"""LLM client abstraction used by the API service."""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from ingestion_job.rag_system.config import ApiConfig


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
        if not all([config.model_api_key, config.model_name, config.model_url]):
            raise RuntimeError(
                "MODEL_API_KEY, MODEL_NAME, MODEL_URL all required"
            )

        url = config.model_url
        if "anthropic" in url:
            self._provider = "anthropic"
        elif "openai" in url:
            self._provider = "openai"
        elif "mistral" in url:
            self._provider = "mistral"
        elif "kimi" in url or "moonshot" in url:
            self._provider = "kimi"
        else:
            raise RuntimeError(f"Unknown provider in MODEL_URL: {url}")

        self.config = config
        self._client = OpenAI(api_key=config.model_api_key, base_url=url)

    def ask(self, prompt: str) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.config.max_answer_tokens,
        )
        return LLMResponse(
            answer=(response.choices[0].message.content or "").strip(),
            provider=self._provider,
            model=self.config.model_name,
        )
