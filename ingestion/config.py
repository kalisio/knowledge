"""Ingestion-job configuration (extends the shared RuntimeConfig)."""

from dataclasses import dataclass, field

from config import RuntimeConfig, env_str, require


@dataclass(frozen=True)
class IngestionConfig(RuntimeConfig):
    """Configuration for the repository indexing job.

    Adds the workspace location, the k-clone target and the logging level on
    top of RuntimeConfig (which pins Qdrant and the embedding model). Built
    lazily via get_ingestion_config(), so importing a module does not read the
    env at import time.
    """

    # Workspace root: where the repositories to index are cloned (k-clone).
    repos_dir: str = field(
        default_factory=lambda: require("KALISIO_DEVELOPMENT_DIR"))

    # k-clone target: which organization/workspace to clone and index.
    kli_organization: str = field(
        default_factory=lambda: env_str("KLI_ORGANIZATION", "kalisio"))
    kli_workspace: str = field(
        default_factory=lambda: env_str("KLI_WORKSPACE", "apps"))

    # Qdrant collection holding the ingestion metadata (last-ingestion cursor).
    qdrant_collection_metadata: str = field(
        default_factory=lambda: require("QDRANT_COLLECTION_METADATA"))

    # Logging verbosity for the "knowledge" logger (stderr).
    log_level: str = field(
        default_factory=lambda: env_str("LOG_LEVEL", "INFO"))


_config = None


# Lazily build and cache the ingestion configuration. Constructed on first use
# (not at import), so importing a module does not require the env to be set yet.
def get_ingestion_config():
    global _config
    if _config is None:
        _config = IngestionConfig()
    return _config
