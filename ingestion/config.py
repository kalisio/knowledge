from dataclasses import dataclass, field
from config import RuntimeConfig, require, env_str, env_int

@dataclass(frozen=True)
class IngestionConfig(RuntimeConfig):
    # See https://github.com/kalisio/development#installation - use for k-clone
    development_dir: str = field(default_factory=lambda: require("DEVELOPMENT_DIR"))
    kli_organization: str = field(default_factory=lambda: env_str("KLI_ORGANIZATION", "kalisio"))
    kli_workspace: str = field(default_factory=lambda: env_str("KLI_WORKSPACE", "apps"))

    # Logging
    log_level: str = field(default_factory=lambda: env_str("LOG_LEVEL", "INFO"))

    # File extensions to index
    supported_file_extensions: str = field(default_factory=lambda: env_str("SUPPORTED_FILE_EXTENSIONS", "md,js,mjs,cjs,vue,json"))

    # File scanning filters
    max_file_size: int = field(default_factory=lambda: env_int("MAX_FILE_SIZE", 100_000))
    ignored_directories: str = field(default_factory=lambda: env_str("IGNORED_DIRECTORIES", ".git,.svn,.hg,node_modules,bower_components,.yarn,.pnpm-store,dist,build,.output,.next,.nuxt,.vite,coverage,.nyc_output,.c8,__pycache__,.cache,.parcel-cache,.turbo,.github,.gitlab,.vscode,.idea"))
    ignored_filenames: str = field(default_factory=lambda: env_str("IGNORED_FILENAMES", "package.json,package-lock.json,CHANGELOG.md,changelog.md,CHANGES.md,LICENSE.md"))
    ignored_file_pattern: str = field(default_factory=lambda: env_str("IGNORED_FILE_PATTERN", r"\.(min|bundle|chunk)\.\w+$|-lock\.json$"))
 

_config = None


# Create & cache the ingestion config
def get_ingestion_config():
    global _config
    if _config is None:
        _config = IngestionConfig()
    return _config
