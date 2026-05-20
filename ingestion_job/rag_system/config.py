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
DEFAULT_LLM_PROMPT = (
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


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
    port: int = field(default_factory=lambda: _env_int("PORT", 8187))
    workers_count: int = field(default_factory=lambda: _env_int("WORKERS_COUNT", 1))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 6))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", ""))
    llm_endpoint: str = field(default_factory=lambda: _env("LLM_ENDPOINT", ""))
    llm_prompt: str = field(default_factory=lambda: _env("LLM_PROMPT", DEFAULT_LLM_PROMPT))
    max_context_chars: int = field(default_factory=lambda: _env_int("MAX_CONTEXT_CHARS", 14000))
    max_answer_tokens: int = field(default_factory=lambda: _env_int("MAX_ANSWER_TOKENS", 1024))


def _kalisio_development_dir() -> Path:
    value = os.getenv("KALISIO_DEVELOPMENT_DIR")
    if not value:
        raise RuntimeError(
            "KALISIO_DEVELOPMENT_DIR is not set. Source the workspace env "
            "(e.g. `source development/workspaces/services/services.sh knowledge`) "
            "before running the ingestion job."
        )
    return Path(value)


def _repo_list() -> list[str]:
    repos = _env_list("REPO_LIST")
    if not repos:
        raise RuntimeError(
            "REPO_LIST is empty. Set it in knowledge.dec.env to the "
            "comma-separated list of Kalisio repositories to ingest "
            "(e.g. REPO_LIST=kdk,kano,kapp,crisis,skeleton,kfs,k2)."
        )
    return repos


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name)
    if not value:
        return None
    return Path(value)


@dataclass(frozen=True)
class IngestionConfig(RuntimeConfig):
    """Ingestion-job configuration driven entirely by environment variables.

    Every operational toggle previously expressed as a CLI flag is sourced
    from `*.dec.env` files via `services.sh`, matching the convention used
    by every other Kalisio service.
    """

    ingestion_root: Path = field(default_factory=_kalisio_development_dir)
    repos: list[str] = field(default_factory=_repo_list)
    incremental: bool = field(default_factory=lambda: _env_bool("INGESTION_INCREMENTAL", True))
    dry_run: bool = field(default_factory=lambda: _env_bool("INGESTION_DRY_RUN", False))
    recreate: bool = field(default_factory=lambda: _env_bool("INGESTION_RECREATE", False))
    no_prune: bool = field(default_factory=lambda: _env_bool("INGESTION_NO_PRUNE", False))
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))
    kli_config: Path | None = field(default_factory=lambda: _optional_path("KLI_CONFIG"))
    kli_dir: Path | None = field(default_factory=lambda: _optional_path("KLI_DIR"))
