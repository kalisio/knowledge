"""Environment-driven configuration shared across the knowledge service.

Values come from the environment (provisioned via the encrypted service
env, sourced by services.sh). Required settings raise when missing, so a
misconfigured deployment fails fast instead of running against fallbacks.
"""

import os
from dataclasses import dataclass, field


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


@dataclass(frozen=True)
class RuntimeConfig:
    """Settings shared by the API service and the ingestion job.

    These pin the vector store and embedding model, which the ingestion
    job (writer) and the API (reader) must agree on for retrieval to work.
    """

    qdrant_url: str = field(default_factory=lambda: require("QDRANT_URL"))
    qdrant_collection: str = field(
        default_factory=lambda: require("QDRANT_COLLECTION"))
    embedding_model: str = field(
        default_factory=lambda: require("EMBEDDING_MODEL"))
    embedding_batch_size: int = field(
        default_factory=lambda: env_int("EMBEDDING_BATCH_SIZE", 8))


_runtime_config = None


# Lazily build and cache the shared RuntimeConfig. Constructed on first use
# (not at import), so a module can import this without the env being set yet.
def get_runtime_config():
    global _runtime_config
    if _runtime_config is None:
        _runtime_config = RuntimeConfig()
    return _runtime_config
