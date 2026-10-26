"""Answers what an agent risks by touching a file, from what the ingestion
job stored about it."""

import api.clients.vectordb as vectordb


# Everything the index knows about `path` in `repository` that bears on
# changing it: the files that historically change with it, how often it
# moves, and why it last did. One retrieve by id.
#
# `indexed` tells a caller which of the two zeroes it is looking at: a file
# nothing is known about, or a question this index cannot answer -- a
# distinction an agent needs before concluding that a refactor is safe. The
# honest answer there is "this index does not know", never "nothing couples
# to it". The co-change partners are capped by the ingestion job.
def get_file_context(repository, path):
    entry = vectordb.get_file_entry(repository, path)
    if entry is None:
        return _empty(repository, path)
    return {
        "repo": repository,
        "path": path,
        "indexed": True,
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
        "cochange_partners": [],
        "churn": 0,
        "commit_history": [],
    }
