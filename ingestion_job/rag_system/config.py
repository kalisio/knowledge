"""Environment-driven configuration for the v1 RAG system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


DEFAULT_COLLECTION = "kalisio_code_v1"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_QUERY_PREFIX = (
    "Instruct: Given a developer question in French or English, retrieve "
    "the relevant Kalisio documentation page or source code file.\nQuery: "
)


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: list[str] | None = None) -> list[str]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return list(default or [])
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class RuntimeConfig:
    """Shared runtime configuration."""

    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", "http://localhost:6333"))
    qdrant_collection: str = field(default_factory=lambda: _env("QDRANT_COLLECTION", DEFAULT_COLLECTION))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    embedding_batch_size: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 8))
    query_prefix: str = field(default_factory=lambda: _env("QUERY_PREFIX", DEFAULT_QUERY_PREFIX))
    index_version: str = field(default_factory=lambda: _env("INDEX_VERSION", "system.v1"))


@dataclass(frozen=True)
class ApiConfig(RuntimeConfig):
    """FastAPI service configuration."""

    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))
    workers_count: int = field(default_factory=lambda: _env_int("WORKERS_COUNT", 1))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 6))
    model_api_key: str = field(default_factory=lambda: _env("MODEL_API_KEY", ""))
    model_name: str = field(default_factory=lambda: _env("MODEL_NAME", ""))
    model_url: str = field(default_factory=lambda: _env("MODEL_URL", ""))
    max_context_chars: int = field(default_factory=lambda: _env_int("MAX_CONTEXT_CHARS", 14000))
    max_answer_tokens: int = field(default_factory=lambda: _env_int("MAX_ANSWER_TOKENS", 1024))
