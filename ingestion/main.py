"""Command-line entry point for the repository indexing job.

Usage:
    python -m ingestion.main                 # index the whole corpus
    python -m ingestion.main <repo> [...]    # index only the named repos

With no arguments, every git repository directly under
KALISIO_DEVELOPMENT_DIR is indexed, except SKIP_REPOS (the Qdrant storage
dir and this service itself, whose docs/experiments would pollute the
corpus). The exact production corpus is still a team decision, so SKIP_REPOS
is the single knob that defines it.

With arguments, only the named repos (directory names under the same root)
are indexed -- handy for (re)indexing one repo at a time.

The selected repositories are chunked, embedded and upserted into the Qdrant
collection the API queries. IngestionConfig is built first so a misconfigured
run fails fast before any embedding work.
"""

import sys
from pathlib import Path

from ingestion.config import IngestionConfig
from ingestion.pipeline import run


# Directories under the workspace root that are not part of the indexed
# corpus: the Qdrant storage dir, and this service itself (its
# docs/experiments would pollute the corpus). Provisional -- the canonical
# production set is still a team decision; adjust this set to change it.
SKIP_REPOS = {"knowledge", "qdrant_data"}


def main(argv=None):
    config = IngestionConfig()
    root = Path(config.repos_dir)

    # TODO incremental ingestion plan:
    #
    # 1. Clone / update repos via k-clone if needed:
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

    repo_dirs = _discover_repos(root)

    print(f"indexing {len(repo_dirs)} repo(s): "
          f"{', '.join(d.name for d in repo_dirs)}")
    count = run(repo_dirs)
    print(f"indexed {count} chunks into '{config.qdrant_collection}'")
    return 0


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# Every git repository directly under `root`, minus SKIP_REPOS.
def _discover_repos(root):
    return [
        path for path in sorted(root.iterdir())
        if path.name not in SKIP_REPOS and (path / ".git").exists()
    ]


if __name__ == "__main__":
    sys.exit(main())
