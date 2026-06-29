"""Recent commit history for a file, feeding the chunk payload's
commit_history.

Runs `git log` on one file and keeps the meaningful commit subjects,
dropping mechanical noise (merges, chore/bump/release, formatting). The
ingestion pipeline attaches the result to a file's chunks so /search and
the LLM context can show how the file recently changed.

Returns an empty list whenever git or the file's history is unavailable
(not a git work tree, untracked file, timeout), so enrichment can never
break ingestion.
"""

import re
import subprocess


# Commit subjects that are mechanical noise rather than meaningful change.
_NOISE = re.compile(
    r"^(merge\b|chore|bump|release|fix lint|fmt|style)", re.IGNORECASE)

# Default number of recent commit subjects kept per file.
MAX_COMMITS = 5


# Recent meaningful commit subjects touching `source_path` (repo-relative),
# newest first. Empty list if git is missing or anything goes wrong.
def get_recent_commits(repo_dir, source_path, limit=MAX_COMMITS):
    if limit <= 0:
        return []
    kept = [s for s in _git_log_subjects(repo_dir, source_path)
            if _is_meaningful(s)]
    return kept[:limit]


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# True if a commit subject carries real meaning (non-empty, not noise).
def _is_meaningful(subject):
    subject = subject.strip()
    return bool(subject) and _NOISE.match(subject) is None


# Subjects of the non-merge commits touching the file, newest first.
# Renames are followed; returns [] on any git error or timeout.
def _git_log_subjects(repo_dir, source_path):
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "log", "--follow", "--no-merges",
             "--format=%s", "--", source_path],
            capture_output=True, text=True, check=False, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return result.stdout.splitlines()
