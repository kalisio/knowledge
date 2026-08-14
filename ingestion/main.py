import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import ingestion.chunking as chunking
from ingestion.config import get_ingestion_config
from ingestion.file_scanner import scan_indexable_files
from ingestion.indexed_file_state import (
    file_key, find_deleted_files, hash_files, load_indexed_file_hashes,
    select_changed_files)
from utils.logging import configure_logging
import utils.embeddings as embeddings
import utils.vectordb as vectordb


def main():
    config = get_ingestion_config()

    # Configure logger
    configure_logging(config.log_level)
    log = logging.getLogger("knowledge.ingestion")

    # Startup summary
    qdrant_collection_metadata_existing = vectordb.check_collection_exists(config.qdrant_collection_metadata)
    qdrant_collection_code_existing = vectordb.check_collection_exists(config.qdrant_collection_code)
    code_collection_length = vectordb.get_code_collection_length() if qdrant_collection_code_existing else 0
    last_ingestion = vectordb.get_last_ingestion() if qdrant_collection_metadata_existing else None # TODO : NEED TO TEST
    log.info("qdrant=%s  model=%s  org=%s  workspace=%s  last_ingestion=%s",
             config.qdrant_url, config.embedding_model, config.kli_organization, config.kli_workspace,
             last_ingestion or "none")
    log.info("metadata collection '%s': %s", config.qdrant_collection_metadata, "exists" if qdrant_collection_metadata_existing else "not found")
    log.info("code collection '%s': %s (%d indexed points)", config.qdrant_collection_code, "exists" if qdrant_collection_code_existing else "not found", code_collection_length)

    # Step 1: Create Qdrant collections if needed
    vectordb.ensure_collection(config.qdrant_collection_metadata,
                               config.qdrant_vector_size_collection_metadata)
    vectordb.ensure_collection(config.qdrant_collection_code,
                               config.qdrant_vector_size_collection_code)

    # Capture the recovery cursor before cloning; persisted only on success
    ingestion_started = datetime.now(timezone.utc).isoformat()

    # Step 2: Clone repositories via k-clone
    try:
        subprocess.run(
            ["bash", "k-clone", config.kli_organization, config.kli_workspace],
            check=True
        )
    except subprocess.CalledProcessError as e:
        log.error("k-clone %s %s failed (exit code %d)", config.kli_organization, config.kli_workspace, e.returncode)
        return 1


    # Step 3: Scan the files and compare them with the existing index
    indexable_files = scan_indexable_files(config.development_dir)
    log.info("files to process: %d", len(indexable_files))

    workspace_root = Path(config.development_dir)
    scanned_file_keys = {file_key(path, workspace_root) for path in indexable_files}
    scanned_file_hashes = hash_files(indexable_files)
    indexed_file_hashes = load_indexed_file_hashes()

    deleted_file_keys = find_deleted_files(indexed_file_hashes, scanned_file_keys)
    for repository, source_path in deleted_file_keys:
        vectordb.delete_file(repository, source_path)
    log.info("files deleted: %d", len(deleted_file_keys))

    previous_index_config = vectordb.get_indexed_config()
    current_index_config = {
        "embedding_model": config.embedding_model,
        "chunking_version": chunking.CHUNKING_VERSION,
    }
    index_config_changed = (previous_index_config is not None
                            and previous_index_config != current_index_config)
    if index_config_changed:
        log.info("indexing config changed (%s -> %s), reindexing every file",
                 previous_index_config, current_index_config)

    # Step 4: Chunk, embed and index the files that changed
    files_to_index = (
        indexable_files if index_config_changed
        else select_changed_files(scanned_file_hashes, workspace_root, indexed_file_hashes))
    chunks_to_index = chunking.chunk_files(files_to_index, workspace_root)
    log.info("files to index: %d (chunks: %d)", len(files_to_index), len(chunks_to_index))
    if files_to_index:
        chunk_vectors = (embeddings.encode_batch([chunk["text"] for chunk in chunks_to_index])
                         if chunks_to_index else [])
        # Drop every reindexed file's chunks, including the files that now
        # yield none, so emptied files leave nothing stale behind.
        for path in files_to_index:
            repository, source_path = file_key(path, workspace_root)
            vectordb.delete_file(repository, source_path)
        if chunks_to_index:
            vectordb.upsert(chunks_to_index, chunk_vectors)
    # TODO incremental: enrich chunks with commit_history.

    # Step 5: Persist the indexing state only after a successful run
    vectordb.set_last_ingestion(
        config.qdrant_collection_metadata, ingestion_started,
        config.embedding_model, chunking.CHUNKING_VERSION)
    return 0


if __name__ == "__main__":
    sys.exit(main())
