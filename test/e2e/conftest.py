"""Fixtures wiring the real pipeline to a throwaway workspace and index.

Nothing here fakes the code under test: the ingestion job, the chunkers, the
payload builder, Qdrant, FastAPI, the routing and the auth layer all run for
real. Only the two external dependencies are replaced -- the embedding model
(a lexical stand-in, so a query really ranks the chunk it talks about) and
the LLM (a recorder, so the prompt it receives can be asserted on).
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

# importlib import mode does not put the test directory on the path, and the
# suite shares helpers.py across its modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest

import api.config as api_config
import api.services.embeddings as api_embeddings
import api.services.llm as llm
import api.services.retrieval as retrieval
import ingestion.config as ingestion_config
import ingestion.logger as ingestion_logger
import ingestion.bin as ingestion_bin
import ingestion.main as ingestion_main
import ingestion.services.embeddings as embeddings
import ingestion.services.vectordb as vectordb
import api.services.vectordb as api_vectordb
from api.main import app
from fastapi.testclient import TestClient

from helpers import (CODE_COLLECTION, METADATA_COLLECTION, VECTOR_SIZE,
                     Workspace, drop_collections, lexical_vector,
                     qdrant_reachable)


# The environment a full deployment sets: ingestion settings, API settings,
# LLM settings. Tests override only what they exercise.
def base_env(root, qdrant_url):
    return {
        # ingestion
        "DEVELOPMENT_DIR": str(root),
        "KLI_ORGANIZATION": "kalisio",
        "KLI_WORKSPACE": "apps",
        "LOG_LEVEL": "DEBUG",
        # qdrant
        "QDRANT_URL": qdrant_url,
        "QDRANT_COLLECTION_CODE": CODE_COLLECTION,
        "QDRANT_COLLECTION_METADATA": METADATA_COLLECTION,
        "QDRANT_VECTOR_SIZE_COLLECTION_CODE": str(VECTOR_SIZE),
        # embeddings
        "EMBEDDING_MODEL": "test-model",
        "EMBEDDING_BATCH_SIZE": "4",
        # api + llm
        "LLM_API_KEY": "test-key",
        "LLM_MODEL": "test-llm",
        "LLM_ENDPOINT": "http://localhost:11434/v1/",
        "KNOWLEDGE_AUTH_ENABLED": "false",
        "TOP_K": "6",
    }


# A SentenceTransformer stand-in. It records every call verbatim -- so a test
# can assert the retrieval instruction reached the model -- while scoring on
# the instruction-free text, which keeps lexical similarity meaningful.
class FakeEmbeddingModel:
    def __init__(self):
        self.calls = []

    def encode(self, sentences, batch_size=None, show_progress_bar=False,
               normalize_embeddings=False):
        self.calls.append(SimpleNamespace(
            sentences=sentences, batch_size=batch_size,
            normalize_embeddings=normalize_embeddings))
        if isinstance(sentences, str):
            return np.array(lexical_vector(_strip_instruction(sentences)))
        return np.array(
            [lexical_vector(_strip_instruction(text)) for text in sentences])


# caplog.text only covers the phase currently running, so a run executed by
# a fixture (setup) would be invisible to a test asserting on its logs. This
# view spans both phases.
class Logs:
    def __init__(self, caplog):
        self._caplog = caplog

    @property
    def text(self):
        setup = "\n".join(record.getMessage()
                          for record in self._caplog.get_records("setup"))
        return f"{setup}\n{self._caplog.text}"

    @property
    def records(self):
        return list(self._caplog.get_records("setup")) + list(
            self._caplog.records)

    # Drop what the current phase has logged so far.
    def clear(self):
        self._caplog.clear()


# Reset every cached singleton the configuration and the clients hide behind,
# then apply `env`. Without this a test inherits the previous test's config.
@pytest.fixture
def configure(monkeypatch):
    def apply(**env):
        for name, value in env.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, str(value))
        monkeypatch.setattr(ingestion_config, "_config", None)
        monkeypatch.setattr(api_config, "_config", None)
        monkeypatch.setattr(vectordb, "_client", None)
        monkeypatch.setattr(api_vectordb, "_client", None)
        monkeypatch.setattr(embeddings, "_model", None)
        monkeypatch.setattr(api_embeddings, "_model", None)
    return apply


# The whole pipeline, ready to run: a git workspace, the real ingestion job
# (k-clone stubbed out -- the workspace is already there), the real chunkers,
# the real Qdrant, and an API client reading the very collection the job
# writes. `run()` executes one ingestion and returns its exit code.
@pytest.fixture
def pipeline(tmp_path, monkeypatch, configure, knowledge_logs):
    # The pipeline runs against the real vector database; without it there is
    # no half-run to fall back on.
    if not qdrant_reachable():
        pytest.skip("needs a running Qdrant (QDRANT_URL)")
    qdrant_url = os.environ["QDRANT_URL"]
    configure(**base_env(tmp_path, qdrant_url))
    drop_collections()

    # One stand-in shared by both services: they must embed with the same
    # model for a query vector and a chunk vector to be comparable.
    # The log level is set by the entry point; reset it so a test that
    # lowered it does not silence the next one.
    ingestion_logger.configure_logging("DEBUG")

    model = FakeEmbeddingModel()
    monkeypatch.setattr(embeddings, "_get_model", lambda: model)
    monkeypatch.setattr(api_embeddings, "_get_model", lambda: model)

    embed_batches = []
    real_encode_batch = embeddings.encode_batch

    def counted_encode_batch(texts):
        texts = list(texts)
        embed_batches.append(len(texts))
        return real_encode_batch(texts)

    monkeypatch.setattr(ingestion_main.embeddings, "encode_batch",
                        counted_encode_batch)

    clone_calls = []
    real_run = subprocess.run

    def run_without_kclone(command, *args, **kwargs):
        if command[:2] == ["bash", "k-clone"]:
            clone_calls.append(command)
            return subprocess.CompletedProcess(command, 0)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(ingestion_main.subprocess, "run", run_without_kclone)

    prompts = []

    def recording_ask(prompt):
        prompts.append(prompt)
        return llm.LLMResponse(
            answer="stub answer", provider="stub", model="stub-model")

    monkeypatch.setattr(retrieval.llm, "ask", recording_ask)

    # k-clone drops each organisation's repositories in its own directory
    # under DEVELOPMENT_DIR, so that is where the workspace lives.
    workspace_root = tmp_path / "kalisio"
    workspace_root.mkdir(exist_ok=True)

    context = SimpleNamespace(
        workspace=Workspace(workspace_root),
        root=workspace_root,
        # DEVELOPMENT_DIR holds the organisation directories; the workspace
        # is the one being scanned.
        development_dir=tmp_path,
        qdrant_url=qdrant_url,
        model=model,
        embed_batches=embed_batches,
        clone_calls=clone_calls,
        prompts=prompts,
        logs=Logs(knowledge_logs),
        # run() is the pipeline; run_from_cli() is what `python -m
        # ingestion.bin` really does, entry point included.
        run=lambda: ingestion_main.run(),
        run_from_cli=lambda: ingestion_bin.main(),
        client=TestClient(app),
    )
    yield context
    drop_collections()


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# Drop the retrieval instruction so the stand-in scores the question itself.
def _strip_instruction(text):
    return text.split("\nQuery: ")[-1]

