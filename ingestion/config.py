"""Configuration of the ingestion job.

Self-contained on purpose: the ingestion job and the API are deployed as two
separate images and share nothing but the Qdrant collections they agree on.
Each reads its own settings, and only the ones it actually needs.
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
class Config:
    """Everything the ingestion job reads from the environment."""

    # Qdrant. The job owns these collections: it creates them, fills them and
    # prunes them.
    qdrant_url: str = field(default_factory=lambda: require("QDRANT_URL"))
    qdrant_collection_code: str = field(default_factory=lambda: require("QDRANT_COLLECTION_CODE"))
    qdrant_collection_metadata: str = field(default_factory=lambda: require("QDRANT_COLLECTION_METADATA"))
    qdrant_vector_size_collection_code: int = field(default_factory=lambda: env_int("QDRANT_VECTOR_SIZE_COLLECTION_CODE", 1024))
    qdrant_vector_size_collection_metadata: int = field(default_factory=lambda: env_int("QDRANT_VECTOR_COLLECTION_METADATA", 1))
    qdrant_last_ingestion_key: str = field(default_factory=lambda: env_str("QDRANT_LAST_INGESTION_KEY", "last_ingestion"))

    # Embedding model: must be the one the API queries with, or a query
    # vector and a chunk vector are not comparable.
    embedding_model: str = field(default_factory=lambda: require("EMBEDDING_MODEL"))
    embedding_batch_size: int = field(default_factory=lambda: env_int("EMBEDDING_BATCH_SIZE", 8))

    # The workspace root: the directory holding one folder per organisation
    # (kalisio/, irsn/, airbus/), each holding the cloned repositories. Both
    # k-clone and the scan work from it.
    # See https://github.com/kalisio/development#installation
    development_dir: str = field(default_factory=lambda: require("DEVELOPMENT_DIR"))
    kli_organization: str = field(default_factory=lambda: env_str("KLI_ORGANIZATION", "kalisio"))
    kli_workspace: str = field(default_factory=lambda: env_str("KLI_WORKSPACE", "apps"))

    # Logging
    log_level: str = field(default_factory=lambda: env_str("LOG_LEVEL", "INFO"))

    # Which repositories to index, by name, comma-separated. Empty means
    # every repository the workspace holds -- what a deployed job does. A
    # developer narrows it down (INDEXED_REPOSITORIES=kdk) to work against
    # one project instead of the whole ecosystem; a name that is not on disk
    # is simply skipped.
    indexed_repositories: str = field(default_factory=lambda: env_str("INDEXED_REPOSITORIES", ""))

    # File extensions to index
    supported_file_extensions: str = field(default_factory=lambda: env_str("SUPPORTED_FILE_EXTENSIONS", "md,js,mjs,cjs,vue,json"))

    # Commit history kept per file (stored once per file, not per chunk).
    # The window slides: older commits drop off, new ones come in. The floor
    # keeps a stable file from ending up with no history at all -- 84% of the
    # kdk files have no commit in the last six months. The depth is an
    # optional cap for a very active file; 0 means no cap.
    commit_history_max_age_days: int = field(default_factory=lambda: env_int("COMMIT_HISTORY_MAX_AGE_DAYS", 180))
    commit_history_min_commits: int = field(default_factory=lambda: env_int("COMMIT_HISTORY_MIN_COMMITS", 5))
    commit_history_depth: int = field(default_factory=lambda: env_int("COMMIT_HISTORY_DEPTH", 0))

    # Chunking. Prose (markdown, JSON) is cut smaller than code: a paragraph
    # carries its meaning in fewer characters than a function does.
    chunk_size: int = field(default_factory=lambda: env_int("CHUNK_SIZE", 500))
    chunk_overlap: int = field(default_factory=lambda: env_int("CHUNK_OVERLAP", 80))
    code_chunk_size: int = field(default_factory=lambda: env_int("CODE_CHUNK_SIZE", 800))
    code_chunk_overlap: int = field(default_factory=lambda: env_int("CODE_CHUNK_OVERLAP", 120))

    # File scanning filters
    max_file_size: int = field(default_factory=lambda: env_int("MAX_FILE_SIZE", 100_000))
    ignored_directories: str = field(default_factory=lambda: env_str("IGNORED_DIRECTORIES", ".git,.svn,.hg,node_modules,bower_components,.yarn,.pnpm-store,dist,build,.output,.next,.nuxt,.vite,coverage,.nyc_output,.c8,__pycache__,.cache,.parcel-cache,.turbo,.github,.gitlab,.vscode,.idea"))
    ignored_filenames: str = field(default_factory=lambda: env_str("IGNORED_FILENAMES", "package.json,package-lock.json,CHANGELOG.md,changelog.md,CHANGES.md,LICENSE.md"))
    ignored_file_pattern: str = field(default_factory=lambda: env_str("IGNORED_FILE_PATTERN", r"\.(min|bundle|chunk)\.\w+$|-lock\.json$"))

    # Per-file entries (commit history). Derived from the code collection by
    # default, so an existing deployment needs no new variable.
    @property
    def qdrant_collection_files(self):
        return env_str("QDRANT_COLLECTION_FILES",
                       f"{self.qdrant_collection_code}_files")


_config = None


# Create & cache the ingestion config.
def get_config():
    global _config
    if _config is None:
        _config = Config()
    return _config
