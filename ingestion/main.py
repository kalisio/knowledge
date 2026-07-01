import logging
import subprocess
import sys
from pathlib import Path

from ingestion.config import get_ingestion_config
from ingestion.file_scanner import scan_indexable_files
from ingestion.git_file_changes import find_file_changes
from utils.logging import configure_logging
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

    # Step 3: Clone repositories via k-clone
    try:
        subprocess.run(
            ["bash", "k-clone", config.kli_organization, config.kli_workspace],
            check=True
        )
    except subprocess.CalledProcessError as e:
        log.error("k-clone %s %s failed (exit code %d)", config.kli_organization, config.kli_workspace, e.returncode)
        return 1


	# Step 4: Get files to process
    if is_first_ingestion:
        files_to_process = scan_indexable_files(config.development_dir)
    else:
        files_to_process = find_file_changes(config.development_dir, last_ingestion)
    log.info("files to process: %d", len(files_to_process))
    
    # is_first_ingestion ?
    # ├─ Yes:
    # │    Take every config.supported_file_extensions in config.development_dir
    # │
    # └─ No:
    #      Use last_ingestion only as a recovery cursor to identify files that
    #      may have changed since the previous successful run.
    #      Example candidate source:
    #          git log --since=<last_ingestion_iso8601> --name-only
    #                  --pretty=format:
    #
    # Result:
    #   files_to_process = files that may need reindexation
    




	
    # 4. Confirm actual content changes with file_sha1
    #
    # For each candidate file:
    #   - Read the current file content.
    #   - Compute file_sha1 from the file content itself.
    #   - Compare it with the file_sha1 already stored in Qdrant for the same
    #     (repository, source_path).
    #   - If the hash is unchanged, skip the file.
    #   - If the hash changed, mark the file for reindexation.
    #
    # The final reindexation decision should rely on file_sha1, not on git log:
    # git history is useful to reduce the scan perimeter and to enrich
    # commit_history, but the hash is the reliable state-based check.
    #
    # 5. Synchronize the vector store
    #
    # For each file marked for reindexation:
    #   - Delete the existing chunks for (repository, source_path) to avoid
    #     stale versions remaining in the collection.
    #   - Re-chunk the current file content.
    #   - Recompute embeddings.
    #   - Upsert the new chunks and metadata into the code collection.
    #
    # 6. Persist ingestion metadata
    #
    # Only after a successful run, update the metadata collection with the new
    # last_ingestion timestamp. Do not update it at job start, otherwise a
    # failed run could move the recovery cursor forward and miss files.
    return 0


if __name__ == "__main__":
    sys.exit(main())
