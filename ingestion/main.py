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

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ingestion.config import IngestionConfig
from ingestion import git_changes


# Directories under the workspace root that are not part of the indexed
# corpus: the Qdrant storage dir, and this service itself (its
# docs/experiments would pollute the corpus). Provisional -- the canonical
# production set is still a team decision; adjust this set to change it.
SKIP_REPOS = {"development", "kli", "knowledge", "qdrant_data"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    from ingestion.pipeline import run
    import utils.vectordb as vectordb

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

    repo_dirs = _ensure_repos(root, config, argv)
    last_ingestion = vectordb.get_last_ingestion()
    candidate_files, stale_files = git_changes.candidate_file_changes(
        repo_dirs, last_ingestion)

    print(f"indexing {len(repo_dirs)} repo(s): "
          f"{', '.join(d.name for d in repo_dirs)}")
    if last_ingestion:
        print(f"last successful ingestion: {last_ingestion}")
    _delete_stale_files(stale_files, vectordb)
    count = run(repo_dirs, candidate_files=candidate_files)
    vectordb.set_last_ingestion(_utc_now_iso8601())
    print(f"indexed {count} chunks into '{config.qdrant_collection}'")
    return 0


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# Ensure the workspace has the development tooling and at least one selected
# target repository. A completely fresh workspace needs development first
# (k-clone and workspace manifests), then k-clone can fetch the real corpus.
def _ensure_repos(root, config, names=None):
    root.mkdir(parents=True, exist_ok=True)
    _ensure_development(root, config.development_repo_url)
    _run_k_clone(root, config.kli_organization, config.kli_workspace)
    repo_dirs = _discover_repos(root, names)
    if not repo_dirs:
        selected = ", ".join(names) if names else "any indexable repository"
        raise RuntimeError(
            f"k-clone completed but did not provide {selected} under {root}")
    return repo_dirs


# Ensure root/development exists; it carries development/scripts/k-clone and
# development/workspaces/<workspace>/<workspace>.js, which k-clone needs.
def _ensure_development(root, repo_url):
    development_dir = root / "development"
    k_clone = development_dir / "scripts" / "k-clone"
    if k_clone.exists():
        return
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(development_dir)],
        check=True)


# Run k-clone from its scripts directory because the script sources its
# sibling .kalisio file using a relative path.
def _run_k_clone(root, organization, workspace):
    scripts_dir = root / "development" / "scripts"
    subprocess.run(
        ["bash", "k-clone", organization, workspace],
        cwd=scripts_dir,
        check=True)


# Every git repository directly under `root`, minus SKIP_REPOS; when names
# are provided, keep only those repositories.
def _discover_repos(root, names=None):
    repo_dirs = [
        path for path in sorted(root.iterdir())
        if path.name not in SKIP_REPOS and (path / ".git").exists()
    ]
    if not names:
        return repo_dirs
    selected = set(names)
    return [path for path in repo_dirs if path.name in selected]


# Build repo -> repo-relative file-path candidates from the last successful
# ingestion timestamp. None means "first run, full scan"; an empty mapping
# means "no git activity since last ingestion".
def _candidate_files(repo_dirs, last_ingestion):
    if not last_ingestion:
        return None
    candidates = {}
    for repo_dir in repo_dirs:
        changed = _git_changed_files_since(repo_dir, last_ingestion)
        if changed:
            candidates[Path(repo_dir).name] = changed
    return candidates


# Delete vector-store chunks for files that disappeared from git history
# (deleted files, or the old side of renames).
def _delete_stale_files(stale_files, vectordb):
    for repository, source_paths in stale_files.items():
        for source_path in source_paths:
            vectordb.delete_file(repository, source_path)


# Repo-relative file paths touched since `since` (ISO 8601), newest activity
# first in git history but returned here as a deduplicated set.
def _git_changed_files_since(repo_dir, since):
    candidates, _ = git_changes.candidate_file_changes([repo_dir], since)
    return candidates.get(Path(repo_dir).name, set())


# Current UTC timestamp in compact ISO 8601 form used by the metadata
# collection, e.g. "2026-06-19T10:35:00Z".
def _utc_now_iso8601():
    return (datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"))


if __name__ == "__main__":
    sys.exit(main())
