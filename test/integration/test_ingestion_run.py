import os
import subprocess
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

import ingestion.main as ingestion_main
import utils.vectordb as vectordb

CODE_COLLECTION = "zz_test_ingestion_code"
METADATA_COLLECTION = "zz_test_ingestion_metadata"
VECTOR_SIZE = 4


# True when Qdrant answers at QDRANT_URL. main() talks to Qdrant directly,
# so every test here is skipped when it is unreachable.
def qdrant_reachable():
    url = os.environ.get("QDRANT_URL")
    if not url:
        return False
    try:
        urllib.request.urlopen(f"{url.rstrip('/')}/healthz", timeout=2)
        return True
    except Exception:
        return False


requires_qdrant = pytest.mark.skipif(
    not qdrant_reachable(), reason="needs a running Qdrant (QDRANT_URL)")


# A workspace with k-clone and embeddings stubbed, pointed at throwaway
# collections; embed_calls records how many chunks each run re-embedded, so
# a test can assert on skip behaviour without inspecting point ids by hand.
@pytest.fixture
def workspace(tmp_path, ingestion_env, monkeypatch):
    embed_calls = []

    def fake_encode_batch(texts):
        texts = list(texts)
        embed_calls.append(len(texts))
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    ingestion_env(
        DEVELOPMENT_DIR=str(tmp_path),
        QDRANT_URL=os.environ["QDRANT_URL"],
        QDRANT_COLLECTION_CODE=CODE_COLLECTION,
        QDRANT_COLLECTION_METADATA=METADATA_COLLECTION,
        QDRANT_VECTOR_SIZE_COLLECTION_CODE=VECTOR_SIZE,
        EMBEDDING_MODEL="test-model",
    )
    real_run = subprocess.run

    def run_without_kclone(command, *args, **kwargs):
        if command[:2] == ["bash", "k-clone"]:
            return None
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(ingestion_main.subprocess, "run", run_without_kclone)
    monkeypatch.setattr(
        ingestion_main.embeddings, "encode_batch", fake_encode_batch)

    yield SimpleNamespace(root=tmp_path, embed_calls=embed_calls)

    if vectordb.check_collection_exists(CODE_COLLECTION):
        vectordb.remove_collection(CODE_COLLECTION)
    if vectordb.check_collection_exists(METADATA_COLLECTION):
        vectordb.remove_collection(METADATA_COLLECTION)


# --- a first run indexes everything and persists its state -----------------

@requires_qdrant
def test_first_run_creates_the_missing_collections(workspace, knowledge_logs):
    # Against a fresh Qdrant neither collection exists yet; a run must create
    # both with the configured vector sizes (metadata uses 1-dim dummy
    # vectors) and log each creation.
    for name in (CODE_COLLECTION, METADATA_COLLECTION):
        if vectordb.check_collection_exists(name):
            vectordb.remove_collection(name)
    _write(workspace.root, "kdk/map/base.js", "export function center () {}\n")

    assert ingestion_main.main() == 0

    assert vectordb.get_collection_vector_size(CODE_COLLECTION) == VECTOR_SIZE
    assert vectordb.get_collection_vector_size(METADATA_COLLECTION) == 1
    assert f"collection '{CODE_COLLECTION}' created" in knowledge_logs.text
    assert f"collection '{METADATA_COLLECTION}' created" in knowledge_logs.text


@requires_qdrant
def test_first_run_indexes_every_file_and_persists_state(workspace):
    _write(workspace.root, "kdk/map/base.js", "export function center () {}\n")
    _write(workspace.root, "kdk/docs/guide.md", "# Guide\n\nSome prose.\n")

    assert ingestion_main.main() == 0

    assert _code_points()
    assert vectordb.get_last_ingestion() is not None
    assert vectordb.get_indexed_config() == {
        "embedding_model": "test-model", "chunking_version": 1}


# --- a second run only touches what actually changed -----------------------

@requires_qdrant
def test_second_run_only_reembeds_the_changed_file(workspace):
    _write(workspace.root, "kdk/map/base.js", "export function center () {}\n")
    _write(workspace.root, "kdk/docs/guide.md", "# Guide\n\nSome prose.\n")
    assert ingestion_main.main() == 0
    first_run_chunks = workspace.embed_calls[-1]

    _write(workspace.root, "kdk/map/base.js", "export function zoom () {}\n")
    assert ingestion_main.main() == 0
    second_run_chunks = workspace.embed_calls[-1]

    # base.js changed, guide.md did not -- strictly fewer chunks re-embedded.
    assert 0 < second_run_chunks < first_run_chunks


@requires_qdrant
def test_second_run_with_nothing_changed_embeds_nothing(workspace):
    _write(workspace.root, "kdk/map/base.js", "export function center () {}\n")
    assert ingestion_main.main() == 0
    calls_after_first_run = len(workspace.embed_calls)

    assert ingestion_main.main() == 0

    # Nothing to embed -> encode_batch is not called at all, not called with
    # an empty list.
    assert len(workspace.embed_calls) == calls_after_first_run


# --- a file removed from disk loses its chunks -----------------------------

@requires_qdrant
def test_deleted_file_removes_its_chunks(workspace):
    _write(workspace.root, "kdk/map/base.js", "export function center () {}\n")
    _write(workspace.root, "kdk/docs/guide.md", "# Guide\n\nSome prose.\n")
    assert ingestion_main.main() == 0

    (workspace.root / "kdk/map/base.js").unlink()
    assert ingestion_main.main() == 0

    remaining = {point.payload["source_path"] for point in _code_points()}
    assert "map/base.js" not in remaining
    assert "docs/guide.md" in remaining


# --- a failing clone aborts the run before it touches the index -------------

@requires_qdrant
def test_a_failed_clone_aborts_the_run_with_exit_code_1(
        workspace, monkeypatch, knowledge_logs):
    _write(workspace.root, "kdk/map/base.js", "export function center () {}\n")

    def failing_kclone(command, *args, **kwargs):
        raise subprocess.CalledProcessError(returncode=3, cmd=command)

    monkeypatch.setattr(ingestion_main.subprocess, "run", failing_kclone)

    assert ingestion_main.main() == 1

    assert "k-clone" in knowledge_logs.text
    assert "exit code 3" in knowledge_logs.text
    assert not _code_points()  # nothing was ingested


# --- non-nominal file contents ----------------------------------------------

@requires_qdrant
def test_an_emptied_file_loses_all_its_chunks(workspace):
    # A file emptied between two runs yields no chunks on reindex; its old
    # chunks must still be dropped.
    _write(workspace.root, "kdk/map/base.js", "export function center () {}\n")
    _write(workspace.root, "kdk/docs/guide.md", "# Guide\n\nSome prose.\n")
    assert ingestion_main.main() == 0

    _write(workspace.root, "kdk/map/base.js", "")
    assert ingestion_main.main() == 0

    remaining = {point.payload["source_path"] for point in _code_points()}
    assert "map/base.js" not in remaining
    assert "docs/guide.md" in remaining


@requires_qdrant
def test_a_non_utf8_file_does_not_break_the_run(workspace):
    # Chunking and hashing read with errors="ignore": a file with invalid
    # UTF-8 bytes is ingested with the bad bytes dropped.
    path = _write(workspace.root, "kdk/map/base.js", "placeholder\n")
    path.write_bytes(b"// caf\xe9 comment\nexport function center () {}\n")
    _git(workspace.root / "kdk", "add", "-A")

    assert ingestion_main.main() == 0

    assert _code_points()


# --- a run never destroys an index it could have reused ---------------------

@requires_qdrant
def test_missing_bookkeeping_does_not_wipe_the_indexed_chunks(workspace):
    # A run killed before its last step leaves the metadata collection empty
    # while the code collection stays fully populated.
    _write(workspace.root, "kdk/map/base.js", "export function center () {}\n")
    _write(workspace.root, "kdk/docs/guide.md", "# Guide\n\nSome prose.\n")
    assert ingestion_main.main() == 0
    indexed_ids = {point.id for point in _code_points()}
    vectordb.remove_collection(METADATA_COLLECTION)

    assert ingestion_main.main() == 0

    # Same points, same ids: nothing was dropped and nothing was re-embedded.
    assert {point.id for point in _code_points()} == indexed_ids
    assert workspace.embed_calls == [len(indexed_ids)]


@requires_qdrant
def test_an_emptied_code_collection_refills_without_a_reset(workspace):
    # The digest comparison reads the code collection itself, so an empty one
    # selects the whole corpus.
    _write(workspace.root, "kdk/map/base.js", "export function center () {}\n")
    assert ingestion_main.main() == 0
    vectordb.remove_collection(CODE_COLLECTION)

    assert ingestion_main.main() == 0

    assert _code_points()


# --- only a vector size change warrants recreating a collection -------------

@requires_qdrant
def test_a_changed_vector_size_recreates_the_code_collection(
        workspace, ingestion_env):
    # Vectors of another dimension cannot be upserted into the old collection.
    _write(workspace.root, "kdk/map/base.js", "export function center () {}\n")
    assert ingestion_main.main() == 0

    ingestion_env(
        DEVELOPMENT_DIR=str(workspace.root),
        QDRANT_URL=os.environ["QDRANT_URL"],
        QDRANT_COLLECTION_CODE=CODE_COLLECTION,
        QDRANT_COLLECTION_METADATA=METADATA_COLLECTION,
        QDRANT_VECTOR_SIZE_COLLECTION_CODE=VECTOR_SIZE + 1,
        EMBEDDING_MODEL="test-model",
    )
    monkeypatch_encode = [[0.1] * (VECTOR_SIZE + 1)]
    ingestion_main.embeddings.encode_batch = (
        lambda texts: monkeypatch_encode * len(list(texts)))

    assert ingestion_main.main() == 0

    assert (vectordb.get_collection_vector_size(CODE_COLLECTION)
            == VECTOR_SIZE + 1)
    assert _code_points()


# --- a changed indexing config forces a full rebuild ------------------------

@requires_qdrant
def test_config_change_forces_a_full_reindex(workspace, ingestion_env):
    _write(workspace.root, "kdk/map/base.js", "export function center () {}\n")
    _write(workspace.root, "kdk/docs/guide.md", "# Guide\n\nSome prose.\n")
    assert ingestion_main.main() == 0
    first_run_chunks = workspace.embed_calls[-1]

    # Nothing on disk changed, but the embedding model did.
    ingestion_env(
        DEVELOPMENT_DIR=str(workspace.root),
        QDRANT_URL=os.environ["QDRANT_URL"],
        QDRANT_COLLECTION_CODE=CODE_COLLECTION,
        QDRANT_COLLECTION_METADATA=METADATA_COLLECTION,
        QDRANT_VECTOR_SIZE_COLLECTION_CODE=VECTOR_SIZE,
        EMBEDDING_MODEL="a-different-model",
    )
    assert ingestion_main.main() == 0
    second_run_chunks = workspace.embed_calls[-1]

    assert second_run_chunks == first_run_chunks
    assert (vectordb.get_indexed_config()["embedding_model"]
            == "a-different-model")


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# Write a workspace-relative file, creating parent directories as needed,
# and track it in its repository -- the scanner only sees tracked files.
def _write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    repo_dir = root / Path(relative).parts[0]
    if not (repo_dir / ".git").exists():
        _git(repo_dir, "init", "-q")
    _git(repo_dir, "add", "-A")
    return path


# Run a git command in a repository, quietly.
def _git(repo_dir, *args):
    subprocess.run(["git", "-C", str(repo_dir), *args],
                   capture_output=True, check=True)


# Every point currently stored in the throwaway code collection.
def _code_points():
    client = vectordb._get_qdrant_client()
    records, _ = client.scroll(
        collection_name=CODE_COLLECTION, limit=1000, with_payload=True,
        with_vectors=False)
    return records
