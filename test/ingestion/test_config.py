import pytest

import ingestion.config as ingestion_config
from ingestion.config import env_int, env_str, require


# --- require: a missing required setting must stop the job ----------------

def test_require_returns_the_value_when_it_is_set(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    assert require("QDRANT_URL") == "http://qdrant:6333"


def test_require_raises_when_the_variable_is_unset(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    with pytest.raises(RuntimeError, match="QDRANT_URL"):
        require("QDRANT_URL")


def test_require_raises_when_the_variable_is_empty(monkeypatch):
    # An unfilled .env entry produces an empty value; treating it as set would
    # start the job against an empty URL.
    monkeypatch.setenv("QDRANT_URL", "")
    with pytest.raises(RuntimeError, match="QDRANT_URL"):
        require("QDRANT_URL")


# --- env_str / env_int: optional settings fall back to the default --------

def test_env_str_returns_the_value_when_it_is_set(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    assert env_str("LOG_LEVEL", "INFO") == "DEBUG"


def test_env_str_falls_back_when_unset_or_empty(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert env_str("LOG_LEVEL", "INFO") == "INFO"
    monkeypatch.setenv("LOG_LEVEL", "")
    assert env_str("LOG_LEVEL", "INFO") == "INFO"


def test_env_int_converts_the_value(monkeypatch):
    monkeypatch.setenv("MAX_FILE_SIZE", "2048")
    value = env_int("MAX_FILE_SIZE", 100)
    assert value == 2048
    assert isinstance(value, int)


def test_env_int_falls_back_when_unset_or_empty(monkeypatch):
    monkeypatch.delenv("MAX_FILE_SIZE", raising=False)
    assert env_int("MAX_FILE_SIZE", 100) == 100
    monkeypatch.setenv("MAX_FILE_SIZE", "")
    assert env_int("MAX_FILE_SIZE", 100) == 100


def test_env_int_raises_on_a_non_numeric_value(monkeypatch):
    # Stopping at startup beats indexing with a nonsense limit.
    monkeypatch.setenv("MAX_FILE_SIZE", "big")
    with pytest.raises(ValueError):
        env_int("MAX_FILE_SIZE", 100)


# --- IngestionConfig: defaults, caching, required settings ----------------

def test_ingestion_config_reads_the_environment(ingestion_env):
    config = ingestion_env(
        DEVELOPMENT_DIR="/srv/development", KLI_WORKSPACE="apps")
    assert config.development_dir == "/srv/development"
    assert config.kli_workspace == "apps"


def test_ingestion_config_defaults_are_applied(ingestion_env):
    config = ingestion_env()
    assert config.kli_organization == "kalisio"
    assert config.log_level == "INFO"
    assert config.max_file_size == 100_000
    assert "md" in config.supported_file_extensions.split(",")


def test_ingestion_config_is_cached(ingestion_env):
    # The whole job reads its settings through this accessor; rebuilding the
    # dataclass per call would re-read an environment mutated mid-run.
    config = ingestion_env()
    assert ingestion_config.get_config() is config


def test_the_config_requires_the_development_dir(
        ingestion_env, monkeypatch):
    ingestion_env()                                 # a complete environment
    monkeypatch.delenv("DEVELOPMENT_DIR")           # minus a required setting
    monkeypatch.setattr(ingestion_config, "_config", None)

    with pytest.raises(RuntimeError, match="DEVELOPMENT_DIR"):
        ingestion_config.get_config()


# --- chunking settings -----------------------------------------------------

def test_the_chunk_sizes_have_defaults(ingestion_env):
    config = ingestion_env()

    assert config.chunk_size == 500
    assert config.chunk_overlap == 80
    assert config.code_chunk_size == 800
    assert config.code_chunk_overlap == 120


def test_the_chunk_sizes_are_overridable(ingestion_env):
    config = ingestion_env(CHUNK_SIZE=250, CHUNK_OVERLAP=25,
                           CODE_CHUNK_SIZE=1200, CODE_CHUNK_OVERLAP=200)

    assert config.chunk_size == 250
    assert config.chunk_overlap == 25
    assert config.code_chunk_size == 1200
    assert config.code_chunk_overlap == 200


def test_prose_is_chunked_smaller_than_code_by_default(ingestion_env):
    config = ingestion_env()

    assert config.chunk_size < config.code_chunk_size
