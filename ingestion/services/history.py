"""The commit history of a file: a sliding window, kept once per file.

Stored on the file entry (services/vectordb), not on every chunk, so real
history costs one copy instead of one per chunk -- eight times less on the
kdk corpus.

WHAT IS KEPT. Commits older than COMMIT_HISTORY_MAX_AGE_DAYS drop off as
time passes, and the commits that landed since the last run come in. The
whole window is rebuilt from git on every run, so there is no stored history
to reconcile when a branch is rebased or force-pushed. A plain window would
empty most of the corpus -- 84% of the kdk files have no commit in the last
six months, and those are exactly the stable files whose "why" is hardest to
guess from the code -- so it has a floor: COMMIT_HISTORY_MIN_COMMITS are
kept however old they are.

Every subject is kept as it was written. Judging which ones "look
meaningful" would throw away most of the history: 30% of the kdk commits are
`wip:` carrying the real subject and its issue number, 3% have no prefix at
all, and prefixes get misspelled.

HOW IT IS READ. One `git log` pass per repository, not per file. Asking git
per file costs ~40 ms each, four minutes over the workspace; one pass builds
the history of every file in kdk in 0.2 s. Renames are followed through the
renames git itself reports (-M), which is a little less eager than
`--follow`: a rename hidden inside a heavily edited commit can be missed,
costing one old subject on a file that moved.
"""

import subprocess
import time

from ingestion.config import get_config

# Field separators in the git log format. ASCII record/unit separators:
# neither can appear in a commit subject or a path.
_RECORD = "\x1e"
_UNIT = "\x1f"


# The commit history of each file, as {(repo, path): [subjects]}, newest
# first. One pass per repository, whatever the number of files asked for.
# `repository_dirs` maps a repository name to where it sits on disk -- the
# workspace nests them one level down per organisation, so the name alone
# does not say where to look.
def collect_commit_history(file_keys, repository_dirs):
    paths_by_repo = {}
    for repo, path in file_keys:
        paths_by_repo.setdefault(repo, []).append(path)
    histories = {}
    for repo, paths in paths_by_repo.items():
        repo_dir = repository_dirs.get(repo)
        commits = _repository_commits(repo_dir) if repo_dir else {}
        for path in paths:
            histories[(repo, path)] = _window(commits.get(path, []))
    return histories


# The commit subjects kept for a single file, newest first. Reads the whole
# repository, so prefer collect_commit_history for more than one file.
def read_history(repo_dir, path):
    return _window(_repository_commits(repo_dir).get(path, []))


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# The commits of the window, plus enough older ones to reach the floor, then
# the optional cap.
def _window(commits):
    config = get_config()
    max_age_days = config.commit_history_max_age_days
    if max_age_days <= 0:
        kept = [subject for _, subject in commits]
    else:
        oldest = time.time() - max_age_days * 86400
        kept = [subject for timestamp, subject in commits
                if timestamp >= oldest]
    if len(kept) < config.commit_history_min_commits:
        kept = [subject
                for _, subject in commits[:config.commit_history_min_commits]]
    depth = config.commit_history_depth
    return kept[:depth] if depth > 0 else kept


# {path: [(timestamp, subject)]} for every file the repository tracks,
# newest first. Merges are left out: they record a branch topology, not a
# change to a file. Empty on any git error, so a broken repository costs
# history, never the run.
def _repository_commits(repo_dir):
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "log", "--no-merges",
             "--name-status", "-M", f"--format={_RECORD}%ct{_UNIT}%s"],
            capture_output=True, text=True, check=False, timeout=300)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    if result.returncode != 0:
        return {}
    return _parse_log(result.stdout)


# Walk the log newest-first, attributing every change to the name the file
# has today: a rename tells us what the file was called further back, so the
# older commits land on the same entry.
def _parse_log(output):
    commits = {}
    renamed_to = {}
    timestamp = subject = None
    for line in output.split("\n"):
        if line.startswith(_RECORD):
            timestamp, subject = _parse_header(line)
            continue
        if not line.strip() or timestamp is None:
            continue
        # Attribute the line even when the commit has no subject: it may
        # carry the rename that tells us what this file used to be called.
        path = _attribute(line, renamed_to)
        if path and subject:
            commits.setdefault(path, []).append((timestamp, subject))
    return commits


# "<RS><timestamp><US><subject>" -> (timestamp, subject).
def _parse_header(line):
    timestamp, _, subject = line[len(_RECORD):].partition(_UNIT)
    try:
        return int(timestamp), subject.strip()
    except ValueError:
        return None, None


# The current name of the file a "<status>\t<path>[\t<path>]" line touches,
# recording what a rename means for the commits still to be read.
def _attribute(line, renamed_to):
    fields = line.split("\t")
    if len(fields) < 2:
        return None
    if fields[0].startswith("R") and len(fields) >= 3:
        before, after = fields[1], fields[2]
        current = renamed_to.get(after, after)
        renamed_to[before] = current
        return current
    return renamed_to.get(fields[-1], fields[-1])
