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
    args = sys.argv[1:] if argv is None else argv
    config = IngestionConfig()
    root = Path(config.repos_dir)

    if args:
        repo_dirs = [root / name for name in args]
        missing = [str(d) for d in repo_dirs if not d.is_dir()]
        if missing:
            print("not a repository directory: " + ", ".join(missing),
                  file=sys.stderr)
            return 1
    else:
        repo_dirs = discover_repos(root)
        if not repo_dirs:
            print(f"no repositories found under {root}", file=sys.stderr)
            return 1

    print(f"indexing {len(repo_dirs)} repo(s): "
          f"{', '.join(d.name for d in repo_dirs)}")
    count = run(repo_dirs)
    print(f"indexed {count} chunks into '{config.qdrant_collection}'")
    return 0


# Every git repository directly under `root`, minus SKIP_REPOS.
def discover_repos(root):
    return [
        path for path in sorted(root.iterdir())
        if path.name not in SKIP_REPOS and (path / ".git").exists()
    ]


if __name__ == "__main__":
    sys.exit(main())
