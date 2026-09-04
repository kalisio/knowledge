"""The import graph, from the workspace on disk to the agent's answer.

The unit tests pin what counts as an edge. What this covers is the journey:
the graph is built from the cloned repositories, stored on the same point as
the commit history, and read back by the API through a single retrieve.

test/data/expected/file_entry.json holds one stored point, verbatim -- what
Qdrant really contains for one file once a run has gone through. Regenerate
it after an intentional change to the payload:

    UPDATE_FILE_ENTRY_REFERENCE=1 pytest test/e2e/test_dependency_graph.py
"""

import json
import os
from pathlib import Path

import pytest

import api.clients.vectordb as api_vectordb
import ingestion.clients.vectordb as vectordb

from helpers import FILES_COLLECTION, read_graph_sample, requires_qdrant

DATA = Path(__file__).resolve().parent.parent / "data"

# Where each source of test/data/graph sits in the workspace. The paths are
# the ones the imports inside those files resolve against, so they are part
# of the fixture, not decoration.
STORE = "kdk/core/client/store.js"
API = "kdk/core/client/api.js"
BARREL = "kdk/core.client.js"
MANIFEST = "kdk/package.json"
COMPONENT = "kano/client/components/KLayerList.vue"

SOURCES = {
    STORE: "store.js",
    API: "api.js",
    BARREL: "core.client.js",
    MANIFEST: "package.json",
    COMPONENT: "KLayerList.vue",
}


# A workspace where one file imports another inside its repository, and a
# second repository imports the first one by package name.
@pytest.fixture
def linked(pipeline):
    for source_path, sample in SOURCES.items():
        pipeline.workspace.commit(source_path, read_graph_sample(sample))
    assert pipeline.run() == 0
    return pipeline


# Ask the API the way an agent does.
def dependents_of(pipeline, source_path):
    repository, path = source_path.split("/", 1)
    response = pipeline.client.post(
        "/dependents", json={"repo": repository, "path": path})
    assert response.status_code == 200
    return response.json()


@pytest.mark.Ingestion
class TestServingTheGraph:
    @requires_qdrant
    def test_an_import_inside_a_repository_reaches_the_api(self, linked):
        answer = dependents_of(linked, STORE)

        # Sorted, and "." sorts before "/": the barrel comes first.
        assert answer["dependents"] == [BARREL, API]
        assert answer["dependent_count"] == 2
        assert answer["indexed"] is True

    @requires_qdrant
    def test_an_import_across_repositories_reaches_the_api(self, linked):
        # The edge no local grep can produce: it takes the manifest of one
        # repository to know what '@kalisio/kdk/core.client' designates.
        answer = dependents_of(linked, BARREL)

        assert COMPONENT in answer["dependents"]

    @requires_qdrant
    def test_the_other_direction_travels_too(self, linked):
        answer = dependents_of(linked, API)

        assert answer["dependencies"] == [STORE]
        assert answer["dependent_count"] == 0

    @requires_qdrant
    def test_a_file_the_corpus_never_saw_is_not_a_file_nobody_imports(
            self, linked):
        indexed = dependents_of(linked, API)
        unknown = dependents_of(linked, "kdk/does/not/exist.js")

        # Both have no dependents; only one of them means anything.
        assert indexed["dependent_count"] == unknown["dependent_count"] == 0
        assert indexed["indexed"] is True
        assert unknown["indexed"] is False

    @requires_qdrant
    def test_the_graph_shares_the_point_that_holds_the_commit_history(
            self, linked):
        # One entry per file carries both, so the API answers either
        # question with a single retrieve by id.
        response = linked.client.post(
            "/search", json={"query": "store", "top_k": 5})
        stored = {(result["repo"], result["path"]): result
                  for result in response.json()}
        repository, path = STORE.split("/", 1)

        assert stored[(repository, path)]["commit_history"]
        assert dependents_of(linked, STORE)["dependents"]

    @requires_qdrant
    def test_the_stored_point_matches_its_reference(self, linked):
        # The payload as Qdrant holds it: one point per file, carrying the
        # digest that drives change detection, the commit history, and both
        # directions of the graph. This is the contract between the two
        # services -- the ingestion job writes it, the API reads it back.
        repository, path = STORE.split("/", 1)
        client = vectordb._get_qdrant_client()
        records = client.retrieve(
            collection_name=FILES_COLLECTION,
            ids=[api_vectordb.file_entry_id(repository, path)],
            with_payload=True, with_vectors=True)
        stored = {"id": str(records[0].id),
                  "vector": records[0].vector,
                  "payload": records[0].payload}

        reference_path = DATA / "expected" / "file_entry.json"
        if os.environ.get("UPDATE_FILE_ENTRY_REFERENCE"):
            reference_path.write_text(
                json.dumps(stored, indent=2, ensure_ascii=False) + "\n")
        assert stored == json.loads(reference_path.read_text())

    @requires_qdrant
    def test_a_removed_import_disappears_from_the_graph(self, linked):
        # The graph is rebuilt whole on every run, which is what lets an
        # edge go away. An incremental rebuild could not: the file at the
        # other end of the edge did not change.
        linked.workspace.commit(API, "export function getUser () { return {} }")

        assert linked.run() == 0

        assert dependents_of(linked, STORE)["dependents"] == [BARREL]
