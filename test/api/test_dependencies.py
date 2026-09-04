"""Answering "who imports this file", from the stored graph.

The caller acts on this: an agent reads dependent_count to decide whether a
change is safe. The two zeroes must therefore never look alike -- a file
nothing imports, and a file the corpus has never seen.
"""

import api.clients.vectordb as vectordb
import api.services.dependencies as dependencies


ENTRY = {
    "repo": "kdk",
    "path": "core/client/store.js",
    "commit_history": ["fix: something"],
    "file_sha1": "abc123",
    "dependents": ["kdk/core/client/api.js", "kano/src/main.js"],
    "dependencies": ["kdk/core/client/events.js"],
}


# Stand in for the stored entry the ingestion job writes.
def stub_entry(monkeypatch, payload):
    monkeypatch.setattr(vectordb, "get_file_entry",
                        lambda repo, path: payload)


def test_the_dependents_are_returned(monkeypatch):
    stub_entry(monkeypatch, ENTRY)

    answer = dependencies.get_dependents("kdk", "core/client/store.js")

    assert answer["dependents"] == ["kdk/core/client/api.js",
                                    "kano/src/main.js"]
    assert answer["dependent_count"] == 2
    assert answer["truncated"] is False
    assert answer["indexed"] is True


def test_the_dependencies_travel_along(monkeypatch):
    # The other direction lets an agent walk up one step without a second
    # call: what this file itself would break against.
    stub_entry(monkeypatch, ENTRY)

    answer = dependencies.get_dependents("kdk", "core/client/store.js")

    assert answer["dependencies"] == ["kdk/core/client/events.js"]


def test_a_file_nobody_imports_is_indexed_with_no_dependents(monkeypatch):
    stub_entry(monkeypatch, {**ENTRY, "dependents": []})

    answer = dependencies.get_dependents("kdk", "core/client/store.js")

    assert answer["dependent_count"] == 0
    assert answer["indexed"] is True


def test_a_file_the_corpus_never_saw_says_so(monkeypatch):
    # Same zero, opposite meaning: nothing can be concluded about a change.
    stub_entry(monkeypatch, None)

    answer = dependencies.get_dependents("kdk", "nowhere.js")

    assert answer["dependent_count"] == 0
    assert answer["indexed"] is False
    assert answer["repo"] == "kdk"
    assert answer["path"] == "nowhere.js"


def test_a_long_list_is_cut_but_the_count_stays_exact(monkeypatch):
    # kdk/core.client.js has 114 dependents, well over a thousand tokens of
    # them. The agent needs to know it is load-bearing, not to read every
    # name.
    monkeypatch.setenv("MAX_DEPENDENTS", "3")
    many = [f"kano/src/file{index}.js" for index in range(10)]
    stub_entry(monkeypatch, {**ENTRY, "dependents": many})

    answer = dependencies.get_dependents("kdk", "core/client/store.js")

    assert answer["dependents"] == many[:3]
    assert answer["dependent_count"] == 10
    assert answer["truncated"] is True


def test_an_entry_written_before_the_graph_existed_says_it_does_not_know(
        monkeypatch):
    # A file entry from an ingestion older than the graph carries a history
    # and a digest, and no graph at all. Reporting it as "no dependents"
    # would tell every agent that every refactor is safe -- and that is the
    # state of the whole corpus between deploying this and the first run
    # that rebuilds it.
    stub_entry(monkeypatch, {"repo": "kdk", "path": "old.js",
                             "commit_history": [], "file_sha1": "aaa"})

    answer = dependencies.get_dependents("kdk", "old.js")

    assert answer["dependents"] == []
    assert answer["indexed"] is False


def test_a_file_the_graph_covers_with_no_dependent_is_indexed(monkeypatch):
    # The other side of the same coin: an empty list that the graph really
    # wrote is an answer, and must not be confused with the case above.
    stub_entry(monkeypatch, {**ENTRY, "dependents": [], "dependencies": []})

    answer = dependencies.get_dependents("kdk", "core/client/store.js")

    assert answer["dependents"] == []
    assert answer["indexed"] is True
