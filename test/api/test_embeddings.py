"""Embedding a question into a query vector.

The API embeds queries, the ingestion job embeds documents, and the two
differ on purpose (see test/ingestion/test_embeddings.py for the other side
and the e2e suite for the asymmetry itself).
"""

import pytest

import api.clients.embeddings as embeddings
from api.clients.embeddings import QUERY_PREFIX


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
        return FakeVector([0.1, 0.2])


@pytest.fixture
def model(monkeypatch):
    recording = RecordingModel()
    monkeypatch.setattr(embeddings, "_get_model", lambda: recording)
    return recording


def test_the_query_carries_the_retrieval_instruction(model):
    # Qwen3-Embedding is trained to embed a query with an instruction and a
    # document without one; dropping it hurts retrieval.
    embeddings.encode("where is the catalog service?")

    payload, _ = model.calls[0]
    assert payload == QUERY_PREFIX + "where is the catalog service?"


def test_the_instruction_names_the_corpus_and_the_languages(model):
    # The instruction is what the model is told to retrieve; it has to
    # describe this corpus, in the languages developers ask in.
    assert "French or English" in QUERY_PREFIX
    assert "Kalisio" in QUERY_PREFIX
    assert QUERY_PREFIX.endswith("Query: ")


def test_the_query_vector_is_normalized(model):
    # Cosine similarity in Qdrant reduces to a dot product only if both
    # sides are L2-normalized.
    embeddings.encode("a question")

    _, kwargs = model.calls[0]
    assert kwargs["normalize_embeddings"] is True


def test_the_progress_bar_is_off(model):
    embeddings.encode("a question")

    _, kwargs = model.calls[0]
    assert kwargs["show_progress_bar"] is False


def test_the_vector_is_a_plain_list(model):
    # Qdrant takes JSON, not a numpy array.
    vector = embeddings.encode("a question")

    assert isinstance(vector, list)
    assert vector == [0.1, 0.2]


def test_the_model_is_loaded_once(monkeypatch):
    loads = []

    class Loader:
        def encode(self, payload, **kwargs):
            return FakeVector([0.1])

    def load(*args, **kwargs):
        loads.append(args)
        return Loader()

    monkeypatch.setattr(embeddings, "_model", None)
    monkeypatch.setattr(embeddings, "SentenceTransformer", load)

    embeddings.encode("first")
    embeddings.encode("second")

    assert len(loads) == 1
