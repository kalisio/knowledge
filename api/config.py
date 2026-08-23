"""Settings the knowledge API reads from the environment."""

import os
from dataclasses import dataclass, field


DEFAULT_SYSTEM_PROMPT = (
    "You are a Kalisio code assistant. Answer the question using only the "
    "provided context from the Kalisio codebase. If the context is "
    "insufficient, say so plainly. Reference the source files you used."
)

DEFAULT_PROMPT_TEMPLATE = """Answer the question based on the context below. \
If the context does not contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""


# Self-contained on purpose: the API and the ingestion job are deployed as
# two separate images and share nothing but the Qdrant collections they
# agree on. Each reads its own settings, and only the ones it needs.
@dataclass(frozen=True)
class Config:
    """Everything the API reads from the environment.

    Built lazily via get_config(), so importing a module does not read the
    environment at import time.
    """

    # Qdrant: read-only. The code collection holds the chunks, the files
    # collection the commit history of each file.
    qdrant_url: str = field(default_factory=lambda: require("QDRANT_URL"))
    qdrant_collection_code: str = field(
        default_factory=lambda: require("QDRANT_COLLECTION_CODE"))

    # Embedding model: must be the one the ingestion job indexed with, or a
    # query vector and a chunk vector are not comparable.
    embedding_model: str = field(
        default_factory=lambda: require("EMBEDDING_MODEL"))

    # LLM provider: required — fail fast if a secret/endpoint is missing.
    llm_api_key: str = field(default_factory=lambda: require("LLM_API_KEY"))
    llm_model: str = field(default_factory=lambda: require("LLM_MODEL"))
    llm_endpoint: str = field(default_factory=lambda: require("LLM_ENDPOINT"))

    # LLM prompts (defaults above). The system instruction is overridable
    # with LLM_PROMPT; prompt_template must contain the {context} and
    # {question} placeholders.
    system_prompt: str = field(
        default_factory=lambda: env_str("LLM_PROMPT", DEFAULT_SYSTEM_PROMPT))
    prompt_template: str = field(default=DEFAULT_PROMPT_TEMPLATE)

    # Retrieval / generation knobs: sensible defaults, env-overridable.
    top_k: int = field(default_factory=lambda: env_int("TOP_K", 6))
    max_context_chars: int = field(
        default_factory=lambda: env_int("MAX_CONTEXT_CHARS", 14000))
    max_answer_tokens: int = field(
        default_factory=lambda: env_int("MAX_ANSWER_TOKENS", 1024))

    # Service binding: 8187 is knowledge's port (team Bruno collection).
    host: str = field(default_factory=lambda: env_str("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: env_int("PORT", 8187))

    # Logging verbosity for the "knowledge" logger (stderr).
    log_level: str = field(
        default_factory=lambda: env_str("LOG_LEVEL", "INFO"))

    # JWT auth. app_secret stays optional so the service can run with auth
    # off; security.py fails closed if it is on while app_secret is missing.
    # Any value but "false" keeps auth enabled.
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

    # Per-file entries (commit history). Derived from the code collection by
    # default, so an existing deployment needs no new variable.
    @property
    def qdrant_collection_files(self):
        return env_str("QDRANT_COLLECTION_FILES",
                       f"{self.qdrant_collection_code}_files")


_config = None


# Lazily build and cache the configuration. Constructed on first use (not at
# import), so importing the API modules does not require every setting to be
# present yet.
def get_config():
    global _config
    if _config is None:
        _config = Config()
    return _config


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------


# Return env var `name`, or raise when it is unset/empty (required setting).
def require(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"required environment variable {name} is not set")
    return value


# Return env var `name` as a string, or `default` when unset/empty.
def env_str(name, default):
    value = os.getenv(name)
    return default if value is None or value == "" else value


# Return env var `name` as an int, or `default` when unset/empty.
def env_int(name, default):
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)
