"""Call the configured LLM to answer a retrieval-augmented prompt.

Speaks to any OpenAI-compatible endpoint (LLM_ENDPOINT): a local server
(Ollama, llama.cpp, LM Studio, ...) or a hosted provider — switching
between them is only a matter of configuration, not code. ask() sends the
configured system instruction (config.system_prompt) telling the model to
answer only from the provided Kalisio context, plus the retrieval-built user
prompt, and returns the answer with the provider and model used.

Config (ApiConfig) and the client are loaded lazily, so importing this
module — or serving /search, which never calls the LLM — does not require
the LLM_* settings to be present.
"""

from collections import namedtuple

from openai import OpenAI

from api.config import get_config


# Shape api/handlers.py consumes: response.answer / .provider / .model.
LLMResponse = namedtuple("LLMResponse", ["answer", "provider", "model"])

_client = None


# Send `prompt` to the configured LLM and return its answer + provenance.
def ask(prompt):
    config = get_config()
    completion = _get_client().chat.completions.create(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=config.max_answer_tokens,
    )
    answer = completion.choices[0].message.content
    return LLMResponse(
        answer=answer,
        provider=_provider(config.llm_endpoint),
        model=config.llm_model,
    )


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------

# Label the provider from the endpoint host, for the response metadata.
def _provider(endpoint):
    host = endpoint.lower()
    if "localhost" in host or "127.0.0.1" in host:
        return "local"
    for name in ("mistral", "openai", "anthropic"):
        if name in host:
            return name
    return "openai-compatible"


# Lazily build the OpenAI-compatible client pointed at LLM_ENDPOINT.
def _get_client():
    global _client
    if _client is None:
        config = get_config()
        _client = OpenAI(
            base_url=config.llm_endpoint, api_key=config.llm_api_key)
    return _client
