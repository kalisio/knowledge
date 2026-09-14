"""Collects what git knows about each indexed file: its commits, and the
files that change with it."""

import subprocess
import time
from collections import Counter

from ingestion.config import get_config

# Field separators in the git log format. ASCII record/unit separators:
# neither can appear in a commit subject or a path.
_RECORD = "\x1e"
_UNIT = "\x1f"

# A commit touching more files than this is a reformat, a rename sweep or
# a dependency bump, not two files evolving together: it is left out of
# the co-change count. Measured on kdk, the pairs that survive are real --
# mixin.base-globe.js and mixin.base-map.js, 31 commits together, no
# import between them.
COCHANGE_MAX_FILES = 20

# How many co-change partners an entry keeps. The strongest ten say what
# to look at; the long tail says nothing.
COCHANGE_PARTNERS = 10


# What git knows about each file, as {(repo, path): {"commit_history",
# "churn", "cochange_partners"}}. One `git log` per repository, whatever
# the number of files asked for. `repository_dirs` maps a repository name
# to where it sits on disk -- the workspace nests them one level down per
# organisation, so the name alone does not say where to look.
#
# commit_history: the subjects kept by the window, newest first.
# churn: how many commits touched the file, over its whole history.
# cochange_partners: the files most often committed together with it, as
# [{"path": "repo/path", "count": n}] -- the coupling no import declares.
def collect_file_history(file_keys, repository_dirs):
    paths_by_repo = {}
    for repo, path in file_keys:
        paths_by_repo.setdefault(repo, []).append(path)
    histories = {}
    for repo, paths in paths_by_repo.items():
        repo_dir = repository_dirs.get(repo)
        commits, groups = _repository_log(repo_dir) if repo_dir else ({}, [])
        partners = _cochange_partners(groups, set(paths))
        for path in paths:
            touched = commits.get(path, [])
            histories[(repo, path)] = {
                "commit_history": _window(touched),
                "churn": len(touched),
                "cochange_partners": [
                    {"path": f"{repo}/{other}", "count": count}
                    for other, count in partners.get(path, [])],
            }
    return histories


# The commit history alone, as {(repo, path): [subjects]}.
def collect_commit_history(file_keys, repository_dirs):
    return {key: entry["commit_history"] for key, entry
            in collect_file_history(file_keys, repository_dirs).items()}


# The commit subjects kept for a single file, newest first. Reads the whole
# repository, so prefer collect_file_history for more than one file.
def read_history(repo_dir, path):
    return _window(_repository_log(repo_dir)[0].get(path, []))


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------


# A plain age window would empty most of the corpus -- 84% of the kdk files
# have no commit in the last six months, and those are exactly the stable
# files whose "why" is hardest to guess from the code -- hence the floor.
# Every subject is kept as it was written: judging which ones look
# meaningful would throw away 30% of the kdk commits, the `wip:` ones that
# carry the real subject and its issue number.
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


# The log of a repository, read once: {path: [(timestamp, subject)]} for
# every file it tracks, newest first, and the set of files each commit
# touched. Merges are left out: they record a branch topology, not a
# change to a file. Empty on any git error, so a broken repository costs
# history, never the run.
def _repository_log(repo_dir):
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "log", "--no-merges",
             "--name-status", "-M", f"--format={_RECORD}%ct{_UNIT}%s"],
            capture_output=True, text=True, check=False, timeout=300)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}, []
    if result.returncode != 0:
        return {}, []
    return _parse_log(result.stdout)


# Walk the log newest-first, attributing every change to the name the file
# has today: a rename tells us what the file was called further back, so the
# older commits land on the same entry. Returns the commits per file and
# the files per commit -- one walk serves both.
def _parse_log(output):
    commits = {}
    groups = []
    renamed_to = {}
    timestamp = subject = None
    for line in output.split("\n"):
        if line.startswith(_RECORD):
            timestamp, subject = _parse_header(line)
            groups.append([])
            continue
        if not line.strip() or timestamp is None:
            continue
        # Attribute the line even when the commit has no subject: it may
        # carry the rename that tells us what this file used to be called.
        path = _attribute(line, renamed_to)
        if path and subject:
            commits.setdefault(path, []).append((timestamp, subject))
            groups[-1].append(path)
    return commits, [group for group in groups if len(group) > 1]


# {path: [(other, count)]} for the files of interest: which other files of
# the repository each one was committed with, strongest first, capped.
def _cochange_partners(groups, wanted):
    pairs = {}
    for group in groups:
        if len(group) > COCHANGE_MAX_FILES:
            continue
        files = set(group)
        for path in files & wanted:
            counter = pairs.setdefault(path, Counter())
            counter.update(files - {path})
    return {path: counter.most_common(COCHANGE_PARTNERS)
            for path, counter in pairs.items()}


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
