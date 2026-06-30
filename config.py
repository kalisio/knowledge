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
    # Qdrant
    qdrant_url: str = field(default_factory=lambda: require("QDRANT_URL"))
    qdrant_collection_code: str = field(default_factory=lambda: require("QDRANT_COLLECTION_CODE"))
    qdrant_vector_size_collection_code: int = field(default_factory=lambda: env_int("QDRANT_VECTOR_SIZE_COLLECTION_CODE", 1024))
    qdrant_vector_size_collection_metadata: int = field(default_factory=lambda: env_int("QDRANT_VECTOR_COLLECTION_METADATA", 1))
    qdrant_collection_metadata: str = field(default_factory=lambda: require("QDRANT_COLLECTION_METADATA"))
    qdrant_last_ingestion_key: str = field(default_factory=lambda: env_str("QDRANT_LAST_INGESTION_KEY", "last_ingestion"))

    # embedding
    embedding_model: str = field(default_factory=lambda: require("EMBEDDING_MODEL"))
    embedding_batch_size: int = field(default_factory=lambda: env_int("EMBEDDING_BATCH_SIZE", 8))


_runtime_config = None


# XXXXXXXXXXXXXXxx
def get_runtime_config():
    global _runtime_config
    if _runtime_config is None:
        _runtime_config = RuntimeConfig()
    return _runtime_config
