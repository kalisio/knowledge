"""FastAPI service configuration (extends the shared RuntimeConfig)."""

import os
from dataclasses import dataclass, field

from config import RuntimeConfig, env_int, env_str, require


@dataclass(frozen=True)
class ApiConfig(RuntimeConfig):
    """Configuration for the /ask and /search API service.

    Adds the LLM provider settings, the retrieval/serving knobs and the JWT
    auth settings the API needs on top of RuntimeConfig. Built lazily via
    get_config(), so importing a module does not read the env at import time.
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

    # JWT auth (mirrors scripts/make-jwt.py). app_secret stays optional so
    # the service can run with auth off; auth.py fails closed if it is on
    # while app_secret is missing. Any value but "false" keeps auth enabled.
    auth_enabled: bool = field(
        default_factory=lambda:
        env_str("KNOWLEDGE_AUTH_ENABLED", "true").lower() != "false")
    app_secret: str | None = field(
        default_factory=lambda: os.getenv("APP_SECRET"))
    jwt_algorithm: str = field(
        default_factory=lambda: env_str("JWT_ALGORITHM", "HS256"))
    jwt_audience: str = field(
        default_factory=lambda: env_str("JWT_AUDIENCE", "kalisio"))
    jwt_issuer: str = field(
        default_factory=lambda: env_str("JWT_ISSUER", "kalisio"))


_config = None


# Lazily build and cache the API configuration. Constructed on first use
# (not at import), so importing the API modules — or serving /search, which
# never calls the LLM — does not require every setting to be present yet.
def get_config():
    global _config
    if _config is None:
        _config = ApiConfig()
    return _config
