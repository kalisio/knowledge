"""Answering "what do I risk touching this file", from the stored entry.

The caller acts on this: an agent reads the co-change partners to decide
whether a change is safe. The two silences must therefore never look alike
-- a file nothing is coupled to, and a file the corpus has never seen.
"""

import api.clients.vectordb as vectordb
import api.services.file_context as file_context


ENTRY = {
    "repo": "kdk",
    "path": "core/client/store.js",
    "commit_history": ["fix: something"],
    "file_sha1": "abc123",
    "churn": 87,
    "cochange_partners": [{"path": "kdk/core/client/api.js", "count": 31}],
}


# Stand in for the stored entry the ingestion job writes.
def stub_entry(monkeypatch, payload):
    monkeypatch.setattr(vectordb, "get_file_entry",
                        lambda repo, path: payload)


def test_the_stored_signals_are_returned(monkeypatch):
    stub_entry(monkeypatch, ENTRY)

    answer = file_context.get_file_context("kdk", "core/client/store.js")

    assert answer["indexed"] is True
    assert answer["churn"] == 87
    assert answer["commit_history"] == ["fix: something"]
    assert answer["cochange_partners"] == [
        {"path": "kdk/core/client/api.js", "count": 31}]


# Coupling no import declares is the whole point of this endpoint: two files
# that keep changing in the same commit are bound even when neither mentions
# the other.
def test_the_cochange_partners_survive(monkeypatch):
    partners = [{"path": "kdk/a.js", "count": 9},
                {"path": "kdk/b.js", "count": 4}]
    stub_entry(monkeypatch, {**ENTRY, "cochange_partners": partners})

    answer = file_context.get_file_context("kdk", "core/client/store.js")

    assert answer["cochange_partners"] == partners


# A file the corpus holds, that simply never moved with anything else.
def test_a_file_coupled_to_nothing_is_still_indexed(monkeypatch):
    stub_entry(monkeypatch, {**ENTRY, "cochange_partners": [], "churn": 0})

    answer = file_context.get_file_context("kdk", "core/client/store.js")

    assert answer["indexed"] is True
    assert answer["cochange_partners"] == []
    assert answer["churn"] == 0


# And a file it has never seen. Same shape, but `indexed` is false: an agent
# must not read this as "nothing is coupled to it".
def test_an_unknown_file_is_not_indexed(monkeypatch):
    stub_entry(monkeypatch, None)

    answer = file_context.get_file_context("kdk", "does/not/exist.js")

    assert answer["indexed"] is False
    assert answer["cochange_partners"] == []
    assert answer["churn"] == 0
    assert answer["commit_history"] == []
    assert answer["repo"] == "kdk"
    assert answer["path"] == "does/not/exist.js"
