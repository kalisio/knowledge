"""Command-line entry point for the repository indexing job.

The selected repositories under KALISIO_DEVELOPMENT_DIR are chunked, embedded
and upserted into the Qdrant collection the API queries. IngestionConfig is
built first so a misconfigured run fails fast before any embedding work.
"""

import logging
import subprocess
import sys
from pathlib import Path

from ingestion.config import get_ingestion_config
from utils.logging import configure_logging
import utils.vectordb as vectordb


def main():
    config = get_ingestion_config()
    configure_logging(config.log_level)
    log = logging.getLogger("knowledge.ingestion")

    # Check & crete qdrant collection
    vectordb.ensure_metadata_collection(config.qdrant_collection_metadata)

    # Clone all reposiroe
    _clone_repositories(config)

    # check if is first ingestion
    last_ingestion = vectordb.get_last_ingestion(
        config.qdrant_collection_metadata)
    is_first_ingestion = last_ingestion is None

    # get last_ingetsion date+
    if is_first_ingestion:
        log.info("first ingestion: indexing the whole corpus")
    else:
        log.info("incremental ingestion since %s", last_ingestion)

    # TODO incremental ingestion plan:
    #
    # 1. verifier que les collection matadta et code de qdrant existe 
    # check_qdfran_collection & creteqdrantcolelction
    # 2. si elle n'existe pas les créer
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
    return 0


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# Clone the repositories to index via k-clone, run from development/scripts
# because k-clone sources its sibling .kalisio with a relative path.
def _clone_repositories(config):
    scripts_dir = Path(config.repos_dir) / "development" / "scripts"
    subprocess.run(
        ["bash", "k-clone", config.kli_organization, config.kli_workspace],
        cwd=scripts_dir,
        check=True,
    )


if __name__ == "__main__":
    sys.exit(main())
