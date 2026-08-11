import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ingestion.config import get_ingestion_config
from ingestion.file_scanner import scan_indexable_files
from ingestion.indexed_file_state import (
    find_deleted_files, load_indexed_file_hashes, select_changed_chunks)
import ingestion.pipeline as pipeline
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
    is_first_ingestion = ( # TODO : NEED TO TEST
        not qdrant_collection_metadata_existing
        or not qdrant_collection_code_existing
        or last_ingestion is None
        or code_collection_length == 0
    )
    log.info("qdrant=%s  model=%s  org=%s  workspace=%s  last_ingestion=%s  first_ingestion=%s",
             config.qdrant_url, config.embedding_model, config.kli_organization, config.kli_workspace,
             last_ingestion or "none", is_first_ingestion)
    log.info("metadata collection '%s': %s", config.qdrant_collection_metadata, "exists" if qdrant_collection_metadata_existing else "not found")
    log.info("code collection '%s': %s (%d indexed points)", config.qdrant_collection_code, "exists" if qdrant_collection_code_existing else "not found", code_collection_length)

    # Step 1: Reset collections on first ingestion
    if is_first_ingestion:
        if qdrant_collection_metadata_existing:
            log.info("resetting collection '%s'", config.qdrant_collection_metadata)
            vectordb.remove_collection(config.qdrant_collection_metadata)
        if qdrant_collection_code_existing:
            log.info("resetting collection '%s'", config.qdrant_collection_code)
            vectordb.remove_collection(config.qdrant_collection_code)

    # Step 2: Create Qdrant collections if needed
    qdrant_collection_metadata_existing = vectordb.check_collection_exists(config.qdrant_collection_metadata)
    qdrant_collection_code_existing = vectordb.check_collection_exists(config.qdrant_collection_code)
    if not qdrant_collection_metadata_existing:
        log.info("creating collection '%s' (vector_size=%d)", config.qdrant_collection_metadata, config.qdrant_vector_size_collection_metadata)
        vectordb.create_collection(config.qdrant_collection_metadata, config.qdrant_vector_size_collection_metadata)
    if not qdrant_collection_code_existing:
        log.info("creating collection '%s' (vector_size=%d)", config.qdrant_collection_code, config.qdrant_vector_size_collection_code)
        vectordb.create_collection(config.qdrant_collection_code, config.qdrant_vector_size_collection_code)

    # Capture the recovery cursor before cloning; persisted only on success
    ingestion_started = datetime.now(timezone.utc).isoformat()

    # Step 3: Clone repositories via k-clone
    try:
        subprocess.run(
            ["bash", "k-clone", config.kli_organization, config.kli_workspace],
            check=True
        )
    except subprocess.CalledProcessError as e:
        log.error("k-clone %s %s failed (exit code %d)", config.kli_organization, config.kli_workspace, e.returncode)
        return 1


	# Step 4: Scan the corpus and compare it against what's indexed
    files_to_process = scan_indexable_files(config.development_dir)
    log.info("files to process: %d", len(files_to_process))

    chunks = pipeline.chunk_files(files_to_process, Path(config.development_dir))
    current_files = {(chunk["metadata"]["repository"], chunk["metadata"]["source_path"])
                      for chunk in chunks}
    indexed_file_hashes = load_indexed_file_hashes()

    deleted_files = find_deleted_files(indexed_file_hashes, current_files)
    for repository, source_path in deleted_files:
        vectordb.delete_file(repository, source_path)
    log.info("files deleted: %d", len(deleted_files))

    indexed_config = vectordb.get_indexed_config()
    current_config = {
        "embedding_model": config.embedding_model,
        "chunking_version": pipeline.CHUNKING_VERSION,
    }
    config_changed = indexed_config is not None and indexed_config != current_config
    if config_changed:
        log.info("indexing config changed (%s -> %s), reindexing every file",
                 indexed_config, current_config)

    # Step 5: Chunk, embed and index the files that changed
    chunks_to_index = (
        chunks if config_changed else select_changed_chunks(chunks, indexed_file_hashes))
    log.info("chunks to index: %d", len(chunks_to_index))
    if chunks_to_index:
        vectors = embeddings.encode_batch([chunk["text"] for chunk in chunks_to_index])
        changed_files = {(chunk["metadata"].get("repository", ""), chunk["metadata"]["source_path"])
                          for chunk in chunks_to_index}
        for repository, source_path in changed_files:
            vectordb.delete_file(repository, source_path)
        vectordb.upsert(chunks_to_index, vectors)
    # TODO incremental: enrich chunks with commit_history.

    # Step 6: Persist the indexing state only after a successful run
    vectordb.set_last_ingestion(
        config.qdrant_collection_metadata, ingestion_started,
        config.embedding_model, pipeline.CHUNKING_VERSION)
    return 0


if __name__ == "__main__":
    sys.exit(main())
