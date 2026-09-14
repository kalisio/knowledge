"""Answers what an agent risks by touching a file, from what the ingestion
job stored about it."""

import api.clients.vectordb as vectordb
from api.config import get_config


# Everything the index knows about `path` in `repository` that bears on
# changing it: the files that import it, the files that historically change
# with it, how often it moves, and why it last did. One retrieve by id.
#
# `indexed` tells a caller which of the two zeroes it is looking at: a file
# nothing depends on, or a question this index cannot answer -- a
# distinction an agent needs before concluding that a refactor is safe.
# Three cases collapse into `indexed: false`, and they collapse on purpose:
# no entry at all, an entry written by an ingestion older than the graph,
# and a file type the graph does not cover. In all three the honest answer
# is "this index does not know", never "nothing depends on it".
#
# The dependents are capped: kdk/core.client.js has 114, well over a
# thousand tokens of them, and an agent that asked one question does not
# need every name to know the file is load-bearing. `dependent_count`
# stays exact. The co-change partners are capped by the ingestion job.
def get_file_context(repository, path):
    entry = vectordb.get_file_entry(repository, path)
    if entry is None or entry.get("dependents") is None:
        return _empty(repository, path)
    dependents = entry["dependents"]
    limit = get_config().max_dependents
    return {
        "repo": repository,
        "path": path,
        "indexed": True,
        "dependents": dependents[:limit],
        "dependent_count": len(dependents),
        "truncated": len(dependents) > limit,
        "dependencies": entry.get("dependencies", []),
        "cochange_partners": entry.get("cochange_partners", []),
        "churn": entry.get("churn", 0),
        "commit_history": entry.get("commit_history", []),
    }


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------


# The answer when the index cannot speak for this file. Same shape as a
# hit, so a caller reads one field rather than branching on a missing key.
def _empty(repository, path):
    return {
        "repo": repository,
        "path": path,
        "indexed": False,
        "dependents": [],
        "dependent_count": 0,
        "truncated": False,
        "dependencies": [],
        "cochange_partners": [],
        "churn": 0,
        "commit_history": [],
    }
