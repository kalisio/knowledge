"""The ingestion pipeline: clone, scan, chunk, embed, index.

Run it with `python -m ingestion.bin`.

The run narrates itself. Indexing a corpus takes minutes, most of them spent
inside a single embedding call, so every step says what it is about to do,
what it found and how long it took. An engineer reading these logs should be
able to answer, without opening the code: what was indexed, what was skipped
and why, where the time went, and what the index holds now.
"""

import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import ingestion.chunkers as chunkers
import ingestion.services.embeddings as embeddings
import ingestion.services.vectordb as vectordb
from ingestion.config import get_config
from ingestion.logger import format_duration, get_logger, step
from ingestion.services.history import collect_commit_history
from ingestion.services.scanner import find_repositories, scan_indexable_files
from ingestion.services.state import (
    find_deleted_files, get_file_key, hash_files, load_indexed_file_hashes,
    select_changed_files)

_STEPS = 7


# Raised when a step leaves the run nothing to continue on. It travels up
# through the step timer, so the failing step is reported as FAILED rather
# than as done.
class IngestionAborted(RuntimeError):
    pass


# Run one ingestion. Returns the process exit code: 0 on success, 1 when a
# step the run cannot recover from failed.
def run():
    config = get_config()
    log = get_logger("ingestion")
    started = time.perf_counter()

    # Without the vector database there is nothing to ingest into.
    try:
        _ingest(config, log, started)
    except (vectordb.QdrantUnreachable, IngestionAborted) as exc:
        log.error("ingestion aborted after %s: %s",
                  format_duration(time.perf_counter() - started), exc)
        return 1
    log.info("ingestion finished in %s",
             format_duration(time.perf_counter() - started))
    return 0


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


def _ingest(config, log, started):
    _log_settings(config, log)

    # Step 1: what is already indexed, and can it be reused?
    with step(log, 1, _STEPS, "inspecting the index"):
        state = _read_index_state(config, log)

    # Step 2: create the collections. A changed indexing config makes every
    # stored chunk unusable -- and chunks written by an older payload
    # contract cannot even be matched for deletion -- so the code collection
    # is recreated rather than updated file by file.
    with step(log, 2, _STEPS, "preparing the collections"):
        vectordb.ensure_collection(
            config.qdrant_collection_metadata,
            config.qdrant_vector_size_collection_metadata)
        vectordb.ensure_collection(
            config.qdrant_collection_code,
            config.qdrant_vector_size_collection_code,
            recreate=state["config_changed"])
        # File entries hold one commit history per file; vectors are dummy.
        vectordb.ensure_collection(
            config.qdrant_collection_files,
            config.qdrant_vector_size_collection_metadata,
            recreate=state["config_changed"])

    # The recovery cursor is captured before cloning and persisted only on
    # success: a failed run must not move it forward and skip files.
    ingestion_started = datetime.now(timezone.utc).isoformat()

    # Step 3: clone the workspace
    with step(log, 3, _STEPS,
              f"cloning the workspace (k-clone {config.kli_organization} "
              f"{config.kli_workspace})"):
        try:
            subprocess.run(
                ["bash", "k-clone", config.kli_organization,
                 config.kli_workspace],
                check=True)
        except subprocess.CalledProcessError as exc:
            log.error("k-clone %s %s failed (exit code %d) -- is the tooling "
                      "on PATH and the git token set?",
                      config.kli_organization, config.kli_workspace,
                      exc.returncode)
            raise IngestionAborted(
                f"k-clone {config.kli_organization} {config.kli_workspace} "
                f"failed with exit code {exc.returncode}") from exc

    # Step 4: scan the workspace and compare it with the index
    with step(log, 4, _STEPS, "scanning the workspace"):
        workspace_root = Path(config.development_dir)
        repositories = {repo.name: repo
                        for repo in find_repositories(workspace_root)}
        indexable_files = scan_indexable_files(workspace_root)
        log.info("  scanning %s%s -- %d repositories, files to process: %d",
                 workspace_root,
                 f" (restricted to {config.indexed_repositories})"
                 if config.indexed_repositories else "",
                 len(repositories), len(indexable_files))
        log.info("  repositories: %s", _repository_list(repositories))
        log.info("  by type: %s", _by_extension(indexable_files))

        scanned_file_keys = {get_file_key(path, workspace_root)
                             for path in indexable_files}
        scanned_file_hashes = hash_files(indexable_files)
        indexed_file_hashes = load_indexed_file_hashes()
        log.info("  already indexed: %d files", len(indexed_file_hashes))

        deleted_file_keys = find_deleted_files(indexed_file_hashes,
                                               scanned_file_keys)
        for repository, path in deleted_file_keys:
            log.debug("  dropping %s/%s", repository, path)
            vectordb.delete_file(repository, path)
            vectordb.delete_file_entry(repository, path)
        log.info("  files deleted: %d", len(deleted_file_keys))

    # Step 5: chunk what changed
    with step(log, 5, _STEPS, "chunking the files that changed"):
        files_to_index = (
            indexable_files if state["config_changed"]
            else select_changed_files(scanned_file_hashes, workspace_root,
                                      indexed_file_hashes))
        _log_change_breakdown(log, files_to_index, workspace_root,
                              indexed_file_hashes, indexable_files, state)
        chunks_to_index = chunkers.chunk_files(files_to_index, workspace_root)
        log.info("  files to index: %d (chunks: %d)",
                 len(files_to_index), len(chunks_to_index))
        if chunks_to_index:
            log.info("  chunks by type: %s", _chunks_by_type(chunks_to_index))
        _log_barren_files(log, files_to_index, chunks_to_index, workspace_root)

    # Step 6: embed and store
    with step(log, 6, _STEPS,
              f"embedding and indexing {len(chunks_to_index)} chunks"):
        if not files_to_index:
            log.info("  nothing changed, nothing to embed")
        else:
            chunk_vectors = (embeddings.encode_batch(
                [chunk["text"] for chunk in chunks_to_index])
                if chunks_to_index else [])
            # Drop every reindexed file's chunks, including the files that
            # now yield none, so emptied files leave nothing stale behind. A
            # recreated collection is already empty.
            if not state["config_changed"]:
                for file_path in files_to_index:
                    repository, path = get_file_key(file_path, workspace_root)
                    vectordb.delete_file(repository, path)
            if chunks_to_index:
                vectordb.upsert(chunks_to_index, chunk_vectors)
                log.info("  %d chunks written to '%s'", len(chunks_to_index),
                         config.qdrant_collection_code)

    # Step 7: refresh the commit history of every scanned file, not only the
    # files being reindexed: the window slides on its own, so a file nobody
    # touched still has to let its oldest commits go.
    with step(log, 7, _STEPS, "refreshing the commit history"):
        histories = collect_commit_history(scanned_file_keys, repositories)
        vectordb.upsert_file_entries(histories)
        subjects = sum(len(entry) for entry in histories.values())
        log.info("  commit histories refreshed: %d (subjects: %d)",
                 len(histories), subjects)
        log.info("  window: %d days, at least %d commits per file",
                 config.commit_history_max_age_days,
                 config.commit_history_min_commits)

    vectordb.set_last_ingestion(
        config.qdrant_collection_metadata, ingestion_started,
        config.embedding_model, chunkers.CHUNKING_VERSION)
    _log_outcome(config, log, len(files_to_index), len(chunks_to_index))


# What the run is configured with, before it does anything.
def _log_settings(config, log):
    log.info("ingestion starting")
    log.info("  qdrant=%s", config.qdrant_url)
    log.info("  collections: code='%s' metadata='%s' files='%s'",
             config.qdrant_collection_code, config.qdrant_collection_metadata,
             config.qdrant_collection_files)
    log.info("  embedding model=%s (vector size %d)", config.embedding_model,
             config.qdrant_vector_size_collection_code)
    log.info("  workspace=%s org=%s kli-workspace=%s%s",
             config.development_dir, config.kli_organization,
             config.kli_workspace,
             f" repositories={config.indexed_repositories}"
             if config.indexed_repositories else "")


# Read what the previous run left behind, and decide whether it can be kept.
def _read_index_state(config, log):
    metadata_exists = vectordb.check_collection_exists(
        config.qdrant_collection_metadata)
    code_exists = vectordb.check_collection_exists(
        config.qdrant_collection_code)
    indexed_points = vectordb.get_code_collection_length() if code_exists else 0
    last_ingestion = vectordb.get_last_ingestion() if metadata_exists else None

    log.info("  metadata collection '%s': %s", config.qdrant_collection_metadata,
             "exists" if metadata_exists else "not found, will be created")
    log.info("  code collection '%s': %s (%d indexed points)",
             config.qdrant_collection_code,
             "exists" if code_exists else "not found, will be created",
             indexed_points)
    log.info("  last_ingestion=%s", last_ingestion or "none")

    previous = vectordb.get_indexed_config() if metadata_exists else None
    current = {
        "embedding_model": config.embedding_model,
        "chunking_version": chunkers.CHUNKING_VERSION,
    }
    changed = previous is not None and previous != current
    if changed:
        log.info("  indexing config changed (%s -> %s), reindexing every file",
                 previous, current)
    elif previous is None:
        log.info("  no previous run recorded: everything will be indexed")
    else:
        log.info("  indexing config unchanged (%s): only what changed will be "
                 "reindexed", current)
    return {"config_changed": changed, "indexed_points": indexed_points}


# Say why each file is being indexed: new, edited, or swept up by a config
# change. Guessing this from a single count is what makes a surprising run
# hard to explain afterwards.
def _log_change_breakdown(log, files_to_index, workspace_root, indexed,
                          scanned, state):
    if state["config_changed"]:
        log.info("  reindexing all %d files (indexing config changed)",
                 len(files_to_index))
        return
    new = sum(1 for path in files_to_index
              if get_file_key(path, workspace_root) not in indexed)
    log.info("  %d new, %d edited, %d unchanged",
             new, len(files_to_index) - new, len(scanned) - len(files_to_index))


# Files that were selected but produced nothing. They are not indexed, so
# nothing records that they were looked at, and they come back as "new" on
# every run -- worth naming rather than leaving as an unexplained count.
def _log_barren_files(log, files_to_index, chunks, workspace_root):
    chunked = {(chunk["metadata"]["repo"], chunk["metadata"]["path"])
               for chunk in chunks}
    barren = [get_file_key(path, workspace_root) for path in files_to_index]
    barren = [key for key in barren if key not in chunked]
    if not barren:
        return
    names = ", ".join(f"{repo}/{path}" for repo, path in barren[:5])
    log.info("  %d file(s) yielded no chunk and stay unindexed "
             "(empty, or a role the chunker skips): %s%s",
             len(barren), names, " ..." if len(barren) > 5 else "")
def _log_outcome(config, log, files, chunks):
    log.info("indexed %d files (%d chunks)", files, chunks)
    try:
        log.info("collection '%s' now holds %d points",
                 config.qdrant_collection_code,
                 vectordb.get_code_collection_length())
    except Exception as exc:                      # never fail on a summary
        log.warning("could not read the collection size back: %s", exc)


# "kdk, kano, crisis (+12 more)" -- enough to spot a missing repository
# without printing two hundred names.
def _repository_list(repositories, limit=8):
    names = sorted(repositories)
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown} (+{len(names) - limit} more)"


# "js: 500, md: 120, vue: 100" -- the shape of what was scanned.
def _by_extension(files):
    counts = Counter(path.suffix.lstrip(".").lower() for path in files)
    return ", ".join(f"{ext}: {count}"
                     for ext, count in counts.most_common()) or "none"


# The same, on the chunks produced: a type yielding far more or far fewer
# chunks than usual is how a broken chunker shows up.
def _chunks_by_type(chunks):
    counts = Counter(Path(chunk["metadata"]["path"]).suffix.lstrip(".").lower()
                     for chunk in chunks)
    return ", ".join(f"{ext}: {count}" for ext, count in counts.most_common())
