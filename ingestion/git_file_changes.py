"""Find repository files changed since the last successful ingestion.

This module reads git history as a recovery cursor: it narrows the scan to
files touched since ``last_ingestion`` and reports old paths that disappeared
so their vector-store chunks can be deleted. It does not decide whether a
current file is truly changed; the pipeline still uses ``file_sha1`` for that
state-based decision.
"""

import subprocess
from pathlib import Path


# Build repo -> repo-relative current candidates and stale paths from git
# history. None candidates means "first run, full scan"; stale paths are files
# deleted or renamed away since the last successful ingestion.
def find_file_changes(repo_dirs, last_ingestion):
    if not last_ingestion:
        return None, {}
    candidates = {}
    stale = {}
    for repo_dir in repo_dirs:
        changed, removed = _file_changes_since(repo_dir, last_ingestion)
        if changed:
            candidates[Path(repo_dir).name] = changed
        if removed:
            stale[Path(repo_dir).name] = removed
    return candidates, stale


# Repo-relative file paths touched since `since`, split into current paths to
# re-check and stale paths to delete from the vector store.
def _file_changes_since(repo_dir, since):
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "log", f"--since={since}",
             "--name-status", "--pretty=format:"],
            capture_output=True, text=True, check=False, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set(), set()
    if result.returncode != 0:
        return set(), set()
    changed = set()
    stale = set()
    for line in result.stdout.splitlines():
        _collect_name_status(line, changed, stale)
    return changed, stale


# Interpret one git --name-status line. Add current file paths to `changed`
# and removed/old paths to `stale`.
def _collect_name_status(line, changed, stale):
    parts = [part.strip().replace("\\", "/")
             for part in line.split("\t") if part.strip()]
    if not parts:
        return
    if len(parts) == 1:
        changed.add(parts[0])
        return
    status = parts[0]
    code = status[0]
    if code == "D" and len(parts) >= 2:
        stale.add(parts[1])
    elif code in {"R", "C"} and len(parts) >= 3:
        if code == "R":
            stale.add(parts[1])
        changed.add(parts[2])
    elif len(parts) >= 2:
        changed.add(parts[1])
