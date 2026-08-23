import logging
import os
import subprocess

import pytest

import api.config as api_config
import ingestion.config as ingestion_config


# The environment both services need to build a configuration. Every test
# runs with it in place -- the chunkers read their sizes from the ingestion
# config, so even a unit test on a chunker needs a valid one. A test
# overrides only the variables it actually exercises.
_BASE_ENV = {
    "DEVELOPMENT_DIR": "/tmp/development",
    "QDRANT_URL": "http://localhost:6333",
    "QDRANT_COLLECTION_CODE": "knowledge_test_code",
    "QDRANT_COLLECTION_METADATA": "knowledge_test_metadata",
    "EMBEDDING_MODEL": "test-model",
    "LLM_API_KEY": "test-key",
    "LLM_MODEL": "test-llm",
    "LLM_ENDPOINT": "http://localhost:11434/v1/",
}


# Put the base environment in place and drop every cached configuration, so
# no test inherits the settings of the one before it.
@pytest.fixture(autouse=True)
def runtime_env(monkeypatch):
    for name, value in _BASE_ENV.items():
        monkeypatch.setenv(name, value)
    _reset_config_caches(monkeypatch)


# One initialised git repository inside a workspace, as k-clone leaves it.
@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "kdk"
    path.mkdir()
    repository = _Repository(path)
    repository.git("init", "-q", "-b", "main")
    repository.git("config", "user.email", "test@kalisio.com")
    repository.git("config", "user.name", "test")
    return repository


# Build an IngestionConfig from a known environment, on top of the base one.
@pytest.fixture
def ingestion_env(monkeypatch):
    def build(**overrides):
        for name, value in {**_BASE_ENV, **overrides}.items():
            monkeypatch.setenv(name, str(value))
        _reset_config_caches(monkeypatch)
        return ingestion_config.get_config()
    return build


# Capture "knowledge.*" log records. configure_logging() disables
# propagation, so caplog's root-logger handler never receives them; the
# handler is attached to the "knowledge" logger directly.
@pytest.fixture
def knowledge_logs(caplog):
    logger = logging.getLogger("knowledge")
    logger.addHandler(caplog.handler)
    yield caplog
    logger.removeHandler(caplog.handler)


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# The two services cache their own configuration; both have to go for a new
# environment to be read.
def _reset_config_caches(monkeypatch):
    monkeypatch.setattr(ingestion_config, "_config", None)
    monkeypatch.setattr(api_config, "_config", None)


# A git repository the tests drive: commits at a chosen date, deletions.
class _Repository:
    def __init__(self, path):
        self.path = path
        self.workspace = path.parent

    # Run a git command in the repository, optionally at a fixed date so
    # tests can order commits deterministically.
    def git(self, *args, date=None):
        env = dict(os.environ)
        if date:
            env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date
        return subprocess.run(
            ["git", "-C", str(self.path), *args],
            capture_output=True, text=True, check=True, env=env)

    # Write a repo-relative file and commit it at `date`.
    def commit(self, path, text, date, message="change"):
        file_path = self.file(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(text)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message, date=date)

    # Delete a repo-relative file and commit the deletion at `date`.
    def remove(self, path, date, message="remove"):
        self.git("rm", "-q", path)
        self.git("commit", "-q", "-m", message, date=date)

    # Absolute path of a repo-relative file.
    def file(self, path):
        return self.path / path
