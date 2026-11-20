"""Cross-lingual retrieval, measured.

A question asked in French must find a passage written in English, and the
other way round. Asserting that the API "returns something" proves nothing
here -- a search always returns its nearest neighbours. What is asserted is
the quality of the ranking, with the measures of information retrieval:
Recall@5, MRR@10 and nDCG@10, condition by condition (EN->EN, FR->EN,
EN->FR, FR->FR, and ES/DE questions over the same corpus).

The corpus (test/data/multilingual/corpus.json) holds every passage in ONE
language only, on close topics, and its questions paraphrase rather than
copy: a passage is found for what it says, across languages, or not at all.

The production path runs for real -- the documents embedded the way the
ingestion job embeds them, the questions the way the API does, with its
retrieval instruction, its translations and its merge. Only Qdrant is
replaced by an exact cosine search over the same vectors.

These tests need the embedding model (1.2 GB) and are skipped when it is not
in the local Hugging Face cache; the translation tests need the Marian
models as well. KNOWLEDGE_TEST_EMBEDDING_MODEL and
KNOWLEDGE_TEST_TRANSLATION_MODELS point them at other models or at local
directories.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

import api.config as api_config
import api.clients.embeddings as api_embeddings
import api.services.retrieval as retrieval
import ingestion.config as ingestion_config
import ingestion.clients.embeddings as ingestion_embeddings

CORPUS = Path(__file__).resolve().parent.parent / "data" / "multilingual" / "corpus.json"

EMBEDDING_MODEL = os.getenv("KNOWLEDGE_TEST_EMBEDDING_MODEL",
                            "Qwen/Qwen3-Embedding-0.6B")
TRANSLATION_MODELS = os.getenv(
    "KNOWLEDGE_TEST_TRANSLATION_MODELS",
    "Helsinki-NLP/opus-mt-fr-en,Helsinki-NLP/opus-mt-en-fr")


# ---------------------------------------------------------------------------
# METRICS -- binary relevance, one ranking of document ids per question
# ---------------------------------------------------------------------------


def recall_at(ranking, relevant, k):
    return len(set(ranking[:k]) & set(relevant)) / len(relevant)


def reciprocal_rank(ranking, relevant, k=10):
    for position, doc in enumerate(ranking[:k], start=1):
        if doc in relevant:
            return 1 / position
    return 0.0


def ndcg_at(ranking, relevant, k=10):
    dcg = sum(1 / math.log2(position + 1)
              for position, doc in enumerate(ranking[:k], start=1)
              if doc in relevant)
    ideal = sum(1 / math.log2(position + 1)
                for position in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


# The metrics are what every threshold below rests on: pin them first, on
# rankings small enough to be worked out by hand. These run without any
# model.
class TestMetrics:
    def test_recall_counts_the_relevant_documents_in_the_top_k(self):
        assert recall_at(["a", "b", "c"], ["c"], 3) == 1.0
        assert recall_at(["a", "b", "c"], ["c"], 2) == 0.0
        assert recall_at(["a", "b"], ["a", "z"], 5) == 0.5

    def test_reciprocal_rank_is_one_over_the_first_relevant_position(self):
        assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0
        assert reciprocal_rank(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)
        assert reciprocal_rank(["a", "b", "c"], ["z"]) == 0.0

    def test_ndcg_discounts_a_relevant_document_by_its_position(self):
        assert ndcg_at(["a", "b"], ["a"]) == 1.0
        assert ndcg_at(["b", "a"], ["a"]) == pytest.approx(1 / math.log2(3))
        assert ndcg_at(["b", "c"], ["a"]) == 0.0


# ---------------------------------------------------------------------------
# AVAILABILITY
# ---------------------------------------------------------------------------


# Whether `name` -- a hub name or a local directory -- can be loaded without
# the network.
def available(name):
    if Path(name).is_dir():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(name, "config.json"), str)


requires_embedding_model = pytest.mark.skipif(
    not available(EMBEDDING_MODEL),
    reason=f"needs {EMBEDDING_MODEL} in the local Hugging Face cache")

requires_translation_models = pytest.mark.skipif(
    not all(available(name.strip()) for name in TRANSLATION_MODELS.split(",")),
    reason=f"needs {TRANSLATION_MODELS} in the local Hugging Face cache")


# ---------------------------------------------------------------------------
# THE INDEX
# ---------------------------------------------------------------------------


# What the ingestion configuration requires to be built, none of which the
# embedding reads.
_INGESTION_ENV = {
    "DEVELOPMENT_DIR": "/tmp/development",
    "QDRANT_URL": "http://localhost:6333",
    "QDRANT_COLLECTION_CODE": "unused",
    "QDRANT_COLLECTION_METADATA": "unused",
}


# The corpus embedded once for the whole module: 24 passages, the model
# loaded once and shared by the "ingestion" and the "API" side.
@pytest.fixture(scope="module")
def index():
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    # A module fixture runs before the per-test base environment of
    # conftest.py is in place, so it brings the settings it needs itself.
    environment = {**_INGESTION_ENV, "EMBEDDING_MODEL": EMBEDDING_MODEL,
                   "HF_HUB_OFFLINE": "1"}
    saved = {name: os.environ.get(name) for name in environment}
    os.environ.update(environment)
    ingestion_config._config = None
    try:
        documents = corpus["documents"]
        vectors = np.array(ingestion_embeddings.encode_batch(
            [f"{doc['title']}\n{doc['text']}" for doc in documents]),
            dtype=np.float32)
        model = ingestion_embeddings._model
        yield {"corpus": corpus, "vectors": vectors, "model": model}
    finally:
        ingestion_embeddings._model = None
        ingestion_config._config = None
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# Run every question of the corpus through the API's retrieval, against an
# exact cosine search over the index, with `translation_models` configured.
# Returns {condition: {metric: mean, "n": count}}.
def evaluate(index, monkeypatch, translation_models):
    monkeypatch.setenv("EMBEDDING_MODEL", EMBEDDING_MODEL)
    monkeypatch.setenv("QUERY_TRANSLATION_MODEL", translation_models)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(api_config, "_config", None)
    monkeypatch.setattr(api_embeddings, "_model", index["model"])

    documents = index["corpus"]["documents"]
    vectors = index["vectors"]

    def search(vector, top_k=5):
        scores = vectors @ np.asarray(vector, dtype=np.float32)
        order = np.argsort(-scores)[:top_k]
        return [{"repo": "test", "path": documents[i]["id"],
                 "chunk_index": 0, "score": float(scores[i]), "lines": "",
                 "breadcrumb": "", "content": ""} for i in order]
    monkeypatch.setattr(retrieval.vectordb, "search", search)

    language_of = {doc["id"]: doc["language"] for doc in documents}
    rows = {}
    for query in index["corpus"]["queries"]:
        ranked = retrieval._search_every_phrasing(
            retrieval._phrasings(query["text"]), top_k=10)
        ranking = [chunk["path"] for chunk in ranked]
        relevant = query["relevant"]
        condition = f"{query['language']}->{language_of[relevant[0]]}".upper()
        rows.setdefault(condition, []).append({
            "recall@5": recall_at(ranking, relevant, 5),
            "mrr@10": reciprocal_rank(ranking, relevant),
            "ndcg@10": ndcg_at(ranking, relevant),
        })
    return {condition: {metric: round(sum(r[metric] for r in values)
                                      / len(values), 3)
                        for metric in values[0]} | {"n": len(values)}
            for condition, values in rows.items()}


# The floors each condition must stay above. Measured on this corpus with
# Qwen3-Embedding-0.6B (see the research page "Multilingual retrieval"),
# then lowered by a margin: they catch a model, an instruction or a merge
# that breaks cross-lingual retrieval, not the noise of a CPU kernel.
FLOORS = {
    "EN->EN": {"recall@5": 0.9, "mrr@10": 0.8},
    "FR->FR": {"recall@5": 0.9, "mrr@10": 0.8},
    "FR->EN": {"recall@5": 0.9, "mrr@10": 0.75},
    "EN->FR": {"recall@5": 0.9, "mrr@10": 0.75},
    "ES->EN": {"recall@5": 0.75, "mrr@10": 0.6},
    "ES->FR": {"recall@5": 0.75, "mrr@10": 0.6},
    "DE->EN": {"recall@5": 0.75, "mrr@10": 0.6},
    "DE->FR": {"recall@5": 0.75, "mrr@10": 0.6},
}


def assert_floors(results, floors):
    failures = [
        f"{condition} {metric}={results[condition][metric]:.3f} < {floor}"
        for condition, metrics in floors.items()
        for metric, floor in metrics.items()
        if results[condition][metric] < floor]
    assert not failures, "\n".join(failures) + f"\nall results: {results}"


@pytest.mark.Multilingual
@requires_embedding_model
class TestCrossLingualRetrieval:
    # The embedding model alone, no translation: the floor everything else
    # stands on. A French question must reach an English passage, and an
    # English question a French one, through the shared vector space.
    def test_the_embedding_model_alone_bridges_the_languages(
            self, index, monkeypatch):
        results = evaluate(index, monkeypatch, "")
        print(json.dumps(results, indent=1))
        assert_floors(results, FLOORS)

    # Across languages, the question must not do markedly worse than within
    # one: the gap between FR->EN and EN->EN is the price of the language
    # barrier, and it has to stay small.
    def test_the_language_barrier_costs_little(self, index, monkeypatch):
        results = evaluate(index, monkeypatch, "")
        for crossed, same in (("FR->EN", "EN->EN"), ("EN->FR", "FR->FR")):
            gap = results[same]["mrr@10"] - results[crossed]["mrr@10"]
            assert gap <= 0.15, f"{crossed} trails {same} by {gap:.3f}"

    # Translating the questions both ways must never take a condition below
    # its floor: the original phrasing is always searched too, and the merge
    # keeps the best score each passage reached.
    @requires_translation_models
    def test_translating_both_ways_keeps_every_condition_above_its_floor(
            self, index, monkeypatch):
        results = evaluate(index, monkeypatch, TRANSLATION_MODELS)
        print(json.dumps(results, indent=1))
        assert_floors(results, FLOORS)
