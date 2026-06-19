"""Ingestion-job configuration (extends the shared RuntimeConfig)."""

from dataclasses import dataclass, field

from config import RuntimeConfig, env_int, env_str, require


@dataclass(frozen=True)
class IngestionConfig(RuntimeConfig):

    kli_organization: str = field(
        default_factory=lambda: env_str("KLI_ORGANIZATION", "kalisio"))
    kli_workspace: str = field(
        default_factory=lambda: env_str("KLI_WORKSPACE", "apps"))
    
    supported_extensions: str = field(
        default_factory=lambda: env_str("SUPPORTED_EXTENSIONS", "{".md", ".js", ".mjs", ".cjs", ".vue", ".json"}"))

