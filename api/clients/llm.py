"""Calls the configured LLM to turn the retrieved chunks into an answer."""

from collections import namedtuple

from openai import OpenAI, OpenAIError

from api.config import get_config


# Shape api/handlers.py consumes: response.answer / .provider / .model.
LLMResponse = namedtuple("LLMResponse", ["answer", "provider", "model"])

_client = None


# Raised when the LLM cannot be reached or refuses the call. The API turns
# it into a 503: a provider being down, or a key being wrong, is an outage
# on our side, not a bad request from the caller.
class LLMUnreachable(RuntimeError):
    pass


# Send `prompt` to the configured LLM and return its answer + provenance.
def ask(prompt):
    config = get_config()
    try:
        completion = _get_client().chat.completions.create(
            model=config.llm_model,
            messages=[
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=config.max_answer_tokens,
        )
    except OpenAIError as exc:
        raise LLMUnreachable(
            f"the LLM at {config.llm_endpoint} could not be reached "
            f"({type(exc).__name__}): check LLM_ENDPOINT, LLM_MODEL and "
            f"LLM_API_KEY") from exc
    answer = completion.choices[0].message.content
    return LLMResponse(
        answer=answer,
        provider=_provider(config.llm_endpoint),
        model=config.llm_model,
    )


# ---------------------------------------------------------------------------
# UTILITIES
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
