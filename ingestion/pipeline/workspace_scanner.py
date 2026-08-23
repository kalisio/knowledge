"""Lists the files of the workspace that are worth indexing."""

import re
import subprocess
from pathlib import Path

from ingestion.config import get_config
from ingestion.logger import get_logger

log = get_logger("ingestion")


# Get the files to index under `root`: the files tracked by the git
# repositories found under it, passed through the scan filters.
#
# The workspace holds one directory per organisation -- kalisio/, irsn/,
# airbus/ -- and each of those holds the cloned repositories, so the
# repositories sit two levels down. A repository sitting directly under the
# root is picked up too, which is how a hand-made workspace is laid out.
#
# git is the authority on what is source: each repository's .gitignore keeps
# work artifacts (data/, outputs/, ...) out of the listing, and a vendored
# submodule stays one entry, not its whole tree.
def scan_indexable_files(root):
    config = get_config()
    supported_file_extensions = config.supported_file_extensions.split(",")
    ignored_directories = set(config.ignored_directories.split(","))
    ignored_filenames = config.ignored_filenames.split(",")
    ignored_file_pattern = re.compile(config.ignored_file_pattern)
    kept = []
    for repo_dir in find_repositories(root):
        for tracked in _git_tracked_files(repo_dir):
            path = repo_dir / tracked
            if path.suffix.lstrip(".").lower() not in supported_file_extensions:
                continue
            if ignored_directories.intersection(path.relative_to(repo_dir).parts[:-1]):
                continue
            if path.name in ignored_filenames:
                continue
            if ignored_file_pattern.search(path.name):
                continue
            # Skips gitlinks and files deleted but still in the index too.
            if not path.is_file() or path.stat().st_size > config.max_file_size:
                continue
            kept.append(path)
    return kept


# Every git repository under `root`, at either depth, in a stable order.
# Narrowed to INDEXED_REPOSITORIES when that names any: a name that is not
# on disk is skipped rather than being an error, so a developer can point
# the job at one project without having the whole ecosystem cloned.
def find_repositories(root):
    root = Path(root)
    wanted = _wanted_repositories()
    seen = []
    for pattern in ("*/.git", "*/*/.git"):
        for git_dir in sorted(root.glob(pattern)):
            repository = git_dir.parent
            if wanted and repository.name not in wanted:
                continue
            # A repository nested inside another one (a submodule checked out
            # in place) is already covered by its parent's file listing.
            if not any(repository.is_relative_to(known) for known in seen):
                seen.append(repository)
    return seen


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------


# The repository names the configuration restricts the scan to, if any.
def _wanted_repositories():
    names = get_config().indexed_repositories
    return {name.strip() for name in names.split(",") if name.strip()}


# The repo-relative paths this repository tracks. A directory that only
# looks like a repository -- an uninitialised submodule or a worktree, whose
# .git is a file pointing somewhere that may not exist -- yields nothing
# instead of killing the run.
def _git_tracked_files(repo_dir):
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "ls-files", "-z"],
            capture_output=True, text=True, check=False, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        log.warning("cannot list the files tracked in %s", repo_dir)
        return []
    if result.returncode != 0:
        log.warning("skipping %s: git ls-files failed (%s)",
                    repo_dir, result.stderr.strip().splitlines()[:1])
        return []
    return [line for line in result.stdout.split("\0") if line]
