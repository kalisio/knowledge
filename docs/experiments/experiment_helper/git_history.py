"""Per-file commit history extraction.

Single source of truth for the "significant commit" filter used by both:

* the Qdrant chunk payload (``commit_history`` field, written at ingestion
  time so ``/search`` returns it without extra queries)
* the planned SQLite ``file_commits`` table from issue #8

Both call ``significant_commits()`` so the filter rule never drifts.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


_NOISE_PREFIXES = re.compile(
    r"^(merge[ :]|chore|bump|release|fix lint|fmt|style)",
    re.IGNORECASE,
)


def is_significant(message: str) -> bool:
    """True if the commit subject is not pure mechanical noise."""
    subject = message.strip()
    if not subject:
        return False
    return _NOISE_PREFIXES.match(subject) is None


def significant_commits(
    repo_path: Path,
    file_rel_path: str,
    *,
    limit: int = 10,
) -> list[str]:
    """Return up to ``limit`` significant commit subjects touching the file.

    Subjects are returned newest first. ``--follow`` is used so renames are
    traced. Returns an empty list if the file has no git history (untracked
    or repo not a git work tree).
    """
    if limit <= 0:
        return []
    try:
        completed = subprocess.run(
            [
                "git",
                "-C", str(repo_path),
                "log",
                "--follow",
                "--no-merges",
                "--format=%s",
                "--",
                file_rel_path,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []

    out: list[str] = []
    for line in completed.stdout.splitlines():
        if is_significant(line):
            out.append(line.strip())
            if len(out) >= limit:
                break
    return out
