"""Ingestion-job configuration (extends the shared RuntimeConfig)."""

from dataclasses import dataclass, field

from config import RuntimeConfig, env_int, env_str, require


@dataclass(frozen=True)
class IngestionConfig(RuntimeConfig):

    repos_dir: str = field(
        default_factory=lambda: require("KALISIO_DEVELOPMENT_DIR"))
    development_repo_url: str = field(
        default_factory=lambda: env_str(
            "KALISIO_DEVELOPMENT_REPO",
            "https://github.com/kalisio/development.git"))

    kli_organization: str = field(
        default_factory=lambda: env_str("KLI_ORGANIZATION", "kalisio"))
    kli_workspace: str = field(
        default_factory=lambda: env_str("KLI_WORKSPACE", "apps"))

    log_level: str = field(
        default_factory=lambda: env_str("LOG_LEVEL", "INFO"))

    qdrant_collection_metadata: str = field(
        default_factory=lambda: require("QDRANT_COLLECTION_METADATA"))

    git_history_limit: int = field(
        default_factory=lambda: env_int("GIT_HISTORY_LIMIT", 10))

    # supported_extensions: str = field(
    #     default_factory=lambda: env_str("SUPPORTED_EXTENSIONS", "{".md", ".js", ".mjs", ".cjs", ".vue", ".json"}"))
