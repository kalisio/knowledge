import re
import subprocess
from pathlib import Path

from ingestion.config import get_ingestion_config


# Get the files to index under `root`: the files tracked by the git
# repositories cloned directly under `root`, passed through the scan
# filters. git is the authority on what is source: each repository's
# .gitignore keeps work artifacts (data/, outputs/, ...) out of the
# listing, and a vendored submodule stays one entry, not its whole tree.
def scan_indexable_files(root):
    config = get_ingestion_config()
    supported_file_extensions = config.supported_file_extensions.split(",")
    ignored_directories = set(config.ignored_directories.split(","))
    ignored_filenames = config.ignored_filenames.split(",")
    ignored_file_pattern = re.compile(config.ignored_file_pattern)
    kept = []
    for git_dir in sorted(Path(root).glob("*/.git")):
        repo_dir = git_dir.parent
        for tracked in _git_tracked_files(repo_dir):
            path = repo_dir / tracked
            if path.suffix.lstrip(".").lower() not in supported_file_extensions:
                continue
            if ignored_directories.intersection(path.relative_to(root).parts[:-1]):
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


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# The repo-relative paths this repository tracks.
def _git_tracked_files(repo_dir):
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "ls-files", "-z"],
        capture_output=True, text=True, check=True)
    return [line for line in result.stdout.split("\0") if line]
