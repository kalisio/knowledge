import subprocess


# Default number of recent commit subjects kept per file.
MAX_COMMITS = 5


# Recent commit subjects touching `source_path` (repo-relative), newest first.
# Empty list if git is missing or anything goes wrong.
def get_recent_commits(repo_dir, source_path, limit=MAX_COMMITS):
    if limit <= 0:
        return []
    return _git_log_subjects(repo_dir, source_path, limit)


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# Subjects of the non-merge commits touching the file, newest first.
# Renames are followed; returns [] on any git error or timeout.
def _git_log_subjects(repo_dir, source_path, limit):
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "log", "--follow", "--no-merges",
             f"-n{limit}", "--format=%s", "--", source_path],
            capture_output=True, text=True, check=False, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [s.strip() for s in result.stdout.splitlines() if s.strip()]
