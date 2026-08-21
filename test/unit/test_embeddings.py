from types import SimpleNamespace

import pytest

import utils.embeddings as embeddings


# Stands in for a SentenceTransformer vector: encode() calls .tolist().
class FakeVector(list):
    def tolist(self):
        return list(self)


# Records every encode() call so a test can assert on the exact text and
# keyword arguments the model receives.
class RecordingModel:
    def __init__(self):
        self.calls = []

    def encode(self, payload, **kwargs):
        self.calls.append((payload, kwargs))
        if isinstance(payload, str):
            return FakeVector([0.1, 0.2])
        return [FakeVector([0.1, 0.2]) for _ in payload]


@pytest.fixture
def model(monkeypatch):
    recording = RecordingModel()
    monkeypatch.setattr(embeddings, "_get_model", lambda: recording)
    monkeypatch.setattr(embeddings, "get_runtime_config",
                        lambda: SimpleNamespace(embedding_batch_size=4))
    return recording


# --- the query/document asymmetry -----------------------------------------
# Both integration suites stub encode/encode_batch out, so this is the only
# place the asymmetry is pinned.

def test_encode_prefixes_the_query_with_the_retrieval_instruction(model):
    embeddings.encode("how to zoom the map?")

    (payload, _), = model.calls
    assert payload.startswith(embeddings.QUERY_PREFIX)
    assert payload.endswith("how to zoom the map?")


def test_encode_batch_embeds_documents_without_the_instruction(model):
    embeddings.encode_batch(["export function a () {}", "# Guide"])

    (payload, _), = model.calls
    assert payload == ["export function a () {}", "# Guide"]


def test_the_same_text_embeds_differently_as_query_and_as_document(model):
    # The asymmetry is the contract: adding the prefix to documents, or
    # dropping it from queries, silently degrades retrieval with every test
    # still green -- except this one.
    embeddings.encode("zoom")
    embeddings.encode_batch(["zoom"])

    (query_payload, _), (document_payload, _) = model.calls
    assert query_payload != document_payload[0]


# --- both sides share the vector-space conventions ------------------------

def test_both_sides_normalize_their_vectors(model):
    # Cosine distance in Qdrant reduces to a dot product only if BOTH sides
    # are L2-normalized; one side dropping it skews every score.
    embeddings.encode("q")
    embeddings.encode_batch(["d"])

    for _, kwargs in model.calls:
        assert kwargs["normalize_embeddings"] is True


def test_encode_batch_uses_the_configured_batch_size(model):
    embeddings.encode_batch(["a", "b"])

    (_, kwargs), = model.calls
    assert kwargs["batch_size"] == 4


def test_encode_batch_materializes_a_generator(model):
    # main() may hand over any iterable; a generator must be consumed once,
    # as a list, not exhausted by a stray len() first.
    embeddings.encode_batch(text for text in ["a", "b"])

    (payload, _), = model.calls
    assert payload == ["a", "b"]


# --- plain-list return values ---------------------------------------------

def test_both_sides_return_plain_lists(model):
    # Callers json-serialize and upsert these; numpy arrays would not do.
    query_vector = embeddings.encode("q")
    document_vectors = embeddings.encode_batch(["d"])

    assert type(query_vector) is list
    assert [type(vector) for vector in document_vectors] == [list]
