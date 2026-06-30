from dataclasses import dataclass, field
from config import RuntimeConfig, require, env_str

@dataclass(frozen=True)
class IngestionConfig(RuntimeConfig):
    # See https://github.com/kalisio/development#installation - use for k-clone
    development_dir: str = field(default_factory=lambda: require("DEVELOPMENT_DIR"))
    kli_organization: str = field(default_factory=lambda: env_str("KLI_ORGANIZATION", "kalisio"))
    kli_workspace: str = field(default_factory=lambda: env_str("KLI_WORKSPACE", "apps"))

    # Logging
    log_level: str = field(default_factory=lambda: env_str("LOG_LEVEL", "INFO"))

    # XXXXXXXXXXXX
    supported_file_extensions: str = field(default_factory=lambda: env_str("vue, js, md, etc."))
 

_config = None


# XXXXXXXXXXXX
def get_ingestion_config():
    global _config
    if _config is None:
        _config = IngestionConfig()
    return _config
