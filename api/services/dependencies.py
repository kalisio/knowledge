"""Answers who imports a file, from the graph the ingestion job stored."""

import api.clients.vectordb as vectordb
from api.config import get_config


# The files that import `path` in `repository`, from the graph the last
# ingestion wrote. `indexed` tells a caller which of the two zeroes it is
# looking at: a file nothing imports, or a question this index cannot
# answer -- a distinction an agent needs before concluding that a refactor
# is safe.
#
# Three cases collapse into `indexed: false`, and they collapse on purpose:
# no entry at all, an entry written by an ingestion older than the graph,
# and a file type the graph does not cover. In all three the honest answer
# is "this index does not know", never "nothing depends on it". The middle
# one is not hypothetical: between deploying this and the next nightly run,
# every entry in the corpus is exactly that.
#
# The list is capped: kdk/core.client.js has 114 dependents, well over a
# thousand tokens of them, and an agent that asked one question does not
# need every name to know the file is load-bearing. `dependent_count`
# stays exact.
def get_dependents(repository, path):
    entry = vectordb.get_file_entry(repository, path)
    if entry is None or entry.get("dependents") is None:
        return _empty(repository, path)
    dependents = entry["dependents"]
    limit = get_config().max_dependents
    return {
        "repo": repository,
        "path": path,
        "dependents": dependents[:limit],
        "dependent_count": len(dependents),
        "truncated": len(dependents) > limit,
        "dependencies": entry.get("dependencies", []),
        "indexed": True,
    }


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------


# The answer when the graph cannot speak for this file. Same shape as a hit,
# so a caller reads one field rather than branching on a missing key.
def _empty(repository, path):
    return {
        "repo": repository,
        "path": path,
        "dependents": [],
        "dependent_count": 0,
        "truncated": False,
        "dependencies": [],
        "indexed": False,
    }
