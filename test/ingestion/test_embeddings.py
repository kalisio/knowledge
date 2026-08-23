"""Embedding documents into vectors.

The ingestion job embeds documents, the API embeds queries, and the two
differ on purpose (see test/api/test_embeddings.py for the other side and
the e2e suite for the asymmetry itself).
"""

import pytest

import ingestion.clients.embeddings as embeddings


# Stands in for a SentenceTransformer vector: encode_batch() calls .tolist().
class FakeVector(list):
    def tolist(self):
        return list(self)


# Records every encode() call so a test can assert on the exact texts and
# keyword arguments the model receives.
class RecordingModel:
    def __init__(self):
        self.calls = []

    def encode(self, payload, **kwargs):
        self.calls.append((payload, kwargs))
        return [FakeVector([0.1, 0.2]) for _ in payload]


@pytest.fixture
def model(monkeypatch):
    recording = RecordingModel()
    monkeypatch.setattr(embeddings, "_get_model", lambda: recording)
    return recording


def test_documents_are_embedded_without_any_instruction(model):
    # The retrieval instruction belongs to the query side only.
    embeddings.encode_batch(["a chunk of code", "another chunk"])

    payload, _ = model.calls[0]
    assert payload == ["a chunk of code", "another chunk"]
    assert all("Instruct:" not in text for text in payload)


def test_the_documents_are_embedded_in_one_call(model):
    embeddings.encode_batch(["one", "two", "three"])

    assert len(model.calls) == 1


def test_the_configured_batch_size_is_used(model, ingestion_env):
    ingestion_env(EMBEDDING_BATCH_SIZE=4)

    embeddings.encode_batch(["a", "b"])

    _, kwargs = model.calls[0]
    assert kwargs["batch_size"] == 4


def test_the_document_vectors_are_normalized(model):
    embeddings.encode_batch(["a"])

    _, kwargs = model.calls[0]
    assert kwargs["normalize_embeddings"] is True


def test_a_generator_is_materialized(model):
    # The model needs a sized sequence, not a stream.
    embeddings.encode_batch(text for text in ["a", "b"])

    payload, _ = model.calls[0]
    assert payload == ["a", "b"]


def test_the_vectors_are_plain_lists(model):
    vectors = embeddings.encode_batch(["a", "b"])

    assert vectors == [[0.1, 0.2], [0.1, 0.2]]
    assert all(isinstance(vector, list) for vector in vectors)


def test_embedding_nothing_calls_the_model_with_nothing(model):
    assert embeddings.encode_batch([]) == []
