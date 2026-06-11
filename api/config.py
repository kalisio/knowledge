"""FastAPI service configuration (extends the shared RuntimeConfig)."""

from dataclasses import dataclass, field

from config import RuntimeConfig, env_int, env_str, require


@dataclass(frozen=True)
class ApiConfig(RuntimeConfig):
    """Configuration for the /ask and /search API service.

    Adds the LLM provider settings and the retrieval/serving knobs the API
    needs on top of RuntimeConfig. Auth settings are not here yet: auth.py
    still reads its own env vars (see the note in the response).
    """

    # LLM provider: required — fail fast if a secret/endpoint is missing.
    llm_api_key: str = field(default_factory=lambda: require("LLM_API_KEY"))
    llm_model: str = field(default_factory=lambda: require("LLM_MODEL"))
    llm_endpoint: str = field(
        default_factory=lambda: require("LLM_ENDPOINT"))

    # Retrieval / generation knobs: sensible defaults, env-overridable.
    top_k: int = field(default_factory=lambda: env_int("TOP_K", 6))
    max_context_chars: int = field(
        default_factory=lambda: env_int("MAX_CONTEXT_CHARS", 14000))
    max_answer_tokens: int = field(
        default_factory=lambda: env_int("MAX_ANSWER_TOKENS", 1024))

    # Service binding: 8187 is knowledge's port (team Bruno collection).
    host: str = field(default_factory=lambda: env_str("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: env_int("PORT", 8187))
