"""Command-line entry point for the repository indexing job.

Usage:
    python -m ingestion.main                 # index the whole corpus
    python -m ingestion.main <repo> [...]    # index only the named repos

With no arguments, every selected git repository directly under
KALISIO_DEVELOPMENT_DIR is indexed. Repository preparation and corpus
selection are handled by ingestion.repository_sync.

With arguments, only the named repos (directory names under the same root)
are indexed -- handy for (re)indexing one repo at a time.

The selected repositories are chunked, embedded and upserted into the Qdrant
collection the API queries. IngestionConfig is built first so a misconfigured
run fails fast before any embedding work.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

from ingestion.config import IngestionConfig
from ingestion import git_file_changes
from ingestion import repository_sync
from ingestion.indexing_pipeline import run
import utils.vectordb as vectordb

def main():
    config = IngestionConfig()

    # Check & crete qdrant collection
    qdrant_collection_metadata_existing = _check_qdfran_collection(config.qdrant_collection_metadata)
    qdrant_collection_code_existing = _check_qdfran_collection(config.qdrant_collection_code)
    if (!qdrant_collection_metadata_existing) _create_qdfran_collection(config.qdrant_collection_metadata)
    if (!qdrant_collection_code_existing) _create_qdfran_collection(config.qdrant_collection_code)

    # Clone all reposiroe 
    subprocess.run([ k-clone config.kli_organization config.kli_worspace])

    # check if is first ingestion
    is_first_ingestion = _is_collection_empty(config.qdrant_collection_code) && _is_collection_empty(config.qdrant_collection_metadata)

    # get last_ingetsion date+
    last_ingestyion_date = null
    if (!is_first_ingestion) last_ingestyion_date =  _get_last_ingestion_date(config.qdrant_collection_metadata)
    log error(need to have last ingestion date if its not first ingestion)


    # TODO incremental ingestion plan:
    # 
    # 1. verifier que les collection matadta et code de qdrant existe check_qdfran_collection & creteqdrantcolelction
    #2. si elle n'existe pas les créer 
    # 3. Clone / update repos via k-clone if needed:
    #    k-clone <organization> <workspace|all>
    #
    # 2. Recover the last successful ingestion timestamp
    #
    # Store it in a dedicated metadata collection, separate from the code
    # collection, with a single record such as:
    #   {
    #     "id": "collection_metadata",
    #     "payload": {"last_ingestion": "2026-06-19T10:35:00Z"}
    #   }
    #
    # Dates should be stored and read in ISO 8601 format. Read this value at
    # the beginning of each run. On the first ingestion, the metadata record
    # does not exist yet.
    #
    # 3. Build the candidate file list
    #
    # first_ingestion ?
    # ├─ Yes:
    # │    Scan every supported file in the selected repositories.
    # │
    # └─ No:
    #      Use last_ingestion only as a recovery cursor to identify files that
    #      may have changed since the previous successful run.
    #      Example candidate source:
    #          git log --since=<last_ingestion_iso8601> --name-only
    #                  --pretty=format:
    #
    # Result:
    #   candidate_files = files that may need reindexation
    #
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

    repo_dirs = repository_sync.ensure_repositories(root, config, argv)
    last_ingestion = vectordb.get_last_ingestion(
        config.qdrant_collection_metadata)
    candidate_files, stale_files = git_file_changes.find_file_changes(
        repo_dirs, last_ingestion)

    print(f"indexing {len(repo_dirs)} repo(s): "
          f"{', '.join(d.name for d in repo_dirs)}")
    if last_ingestion:
        print(f"last successful ingestion: {last_ingestion}")
    _delete_stale_files(stale_files, vectordb)
    count = run(repo_dirs, candidate_files=candidate_files)
    vectordb.set_last_ingestion(
        config.qdrant_collection_metadata, _utc_now_iso8601())
    print(f"indexed {count} chunks into '{config.qdrant_collection_code}'")
    return 0


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# Delete vector-store chunks for files that disappeared from git history
# (deleted files, or the old side of renames).
def _delete_stale_files(stale_files, vectordb):
    for repository, source_paths in stale_files.items():
        for source_path in source_paths:
            vectordb.delete_file(repository, source_path)

# Current UTC timestamp in compact ISO 8601 form used by the metadata
# collection, e.g. "2026-06-19T10:35:00Z".
def _utc_now_iso8601():
    return (datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"))


if __name__ == "__main__":
    sys.exit(main())
