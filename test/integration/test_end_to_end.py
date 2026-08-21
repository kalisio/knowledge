import hashlib
import os
import subprocess
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api.auth as auth
import api.handlers as handlers
import ingestion.main as ingestion_main
import utils.vectordb as vectordb
from api.app import app
from test_ingestion_run import _write, requires_qdrant

CODE_COLLECTION = "zz_test_e2e_code"
METADATA_COLLECTION = "zz_test_e2e_metadata"
VECTOR_SIZE = 4
QUERY_VECTOR = [0.1, 0.2, 0.3, 0.4]

# One file per supported type, across two repositories.
FILES = {
    "kdk/docs/guide.md": "# Guide\n\n## Install\n\nRun yarn install.\n",
    "kdk/map/base.js": "export function center () {\n  return [0, 0]\n}\n",
    "kano/components/KMap.vue": (
        "<template>\n  <div class=\"map\" />\n</template>\n\n"
        "<script>\nexport default { name: 'KMap' }\n</script>\n"),
    "kano/i18n/en.json": (
        '{\n  "KMap": {"TITLE": "Map", "ZOOM": "Zoom in"}\n}\n'),
}


# The full pipeline wired to throwaway collections: ingestion writes them,
# the API (auth disabled, embeddings and LLM stubbed) reads them. Every
# chunk gets the same vector as the stubbed query embedding, so a search
# with a large top_k retrieves the entire corpus.
@pytest.fixture
def corpus(tmp_path, ingestion_env, monkeypatch):
    embed_calls = []

    def fake_encode_batch(texts):
        texts = list(texts)
        embed_calls.append(len(texts))
        return [QUERY_VECTOR for _ in texts]

    ingestion_env(
        DEVELOPMENT_DIR=str(tmp_path),
        QDRANT_URL=os.environ["QDRANT_URL"],
        QDRANT_COLLECTION_CODE=CODE_COLLECTION,
        QDRANT_COLLECTION_METADATA=METADATA_COLLECTION,
        QDRANT_VECTOR_SIZE_COLLECTION_CODE=VECTOR_SIZE,
        EMBEDDING_MODEL="test-model",
    )
    for name in (CODE_COLLECTION, METADATA_COLLECTION):
        if vectordb.check_collection_exists(name):
            vectordb.remove_collection(name)

    real_run = subprocess.run

    def run_without_kclone(command, *args, **kwargs):
        if command[:2] == ["bash", "k-clone"]:
            return None
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(ingestion_main.subprocess, "run", run_without_kclone)
    monkeypatch.setattr(
        ingestion_main.embeddings, "encode_batch", fake_encode_batch)

    monkeypatch.setattr(auth, "get_config", lambda: SimpleNamespace(
        auth_enabled=False, app_secret=None, jwt_algorithm="HS256",
        jwt_audience="kalisio", jwt_issuer="kalisio"))
    monkeypatch.setattr(handlers, "get_config", lambda: SimpleNamespace(
        top_k=50, max_context_chars=14000,
        prompt_template="Context:\n{context}\n\nQuestion: {question}"))
    monkeypatch.setattr(
        handlers.embeddings, "encode", lambda text: QUERY_VECTOR)
    monkeypatch.setattr(handlers.llm, "ask", lambda prompt: SimpleNamespace(
        answer="stub answer", provider="stub", model="stub-model"))

    for source_path, text in FILES.items():
        _write(tmp_path, source_path, text)

    yield SimpleNamespace(
        root=tmp_path, embed_calls=embed_calls, client=TestClient(app))

    for name in (CODE_COLLECTION, METADATA_COLLECTION):
        if vectordb.check_collection_exists(name):
            vectordb.remove_collection(name)


# --- first run: every file type is indexed with the full payload contract --

@requires_qdrant
def test_first_run_ingests_all_four_file_types(corpus):
    assert ingestion_main.main() == 0

    points = _points_by_file()
    assert set(points) == {_key(source_path) for source_path in FILES}
    for source_path, text in FILES.items():
        chunks = [point.payload for point in points[_key(source_path)]]
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        assert {chunk["file_sha1"] for chunk in chunks} == {digest}
        assert {chunk["file_type"] for chunk in chunks} == {
            source_path.rsplit(".", 1)[-1]}
        assert (sorted(chunk["chunk_index"] for chunk in chunks)
                == list(range(len(chunks))))
        assert all(chunk["text"].strip() for chunk in chunks)


@requires_qdrant
def test_ingested_corpus_is_served_by_search_and_ask(corpus):
    assert ingestion_main.main() == 0

    response = corpus.client.post(
        "/search", json={"query": "map", "top_k": 50})
    assert response.status_code == 200
    served = {result["source_path"] for result in response.json()["results"]}
    assert served == {"docs/guide.md", "map/base.js",
                      "components/KMap.vue", "i18n/en.json"}

    response = corpus.client.post("/ask", json={"question": "how to zoom?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "stub answer"
    assert body["sources"]


# --- a query before any ingestion returns the 503 hint ---------------------

@requires_qdrant
def test_querying_before_ingestion_returns_a_503_hint(corpus):
    response = corpus.client.post("/search", json={"query": "map", "top_k": 5})

    assert response.status_code == 503
    assert "ingestion" in response.json()["detail"]


# --- modify one file: exactly that file is reindexed ------------------------

@requires_qdrant
def test_modifying_one_file_reindexes_only_that_file(corpus):
    assert ingestion_main.main() == 0
    before = _points_by_file()
    new_text = "export function zoom () {\n  return 12\n}\n"
    _write(corpus.root, "kdk/map/base.js", new_text)

    assert ingestion_main.main() == 0

    after = _points_by_file()
    changed = after[("kdk", "map/base.js")]
    digest = hashlib.sha1(new_text.encode("utf-8")).hexdigest()
    assert {point.payload["file_sha1"] for point in changed} == {digest}
    # Point ids embed the content digest: no old chunk survives the change.
    assert {point.id for point in changed}.isdisjoint(
        {point.id for point in before[("kdk", "map/base.js")]})
    # Every other file keeps its exact points; only the changed file's
    # chunks were re-embedded.
    for key in set(before) - {("kdk", "map/base.js")}:
        assert ({point.id for point in after[key]}
                == {point.id for point in before[key]})
    assert corpus.embed_calls[-1] == len(changed)

    # The API serves the updated content.
    response = corpus.client.post(
        "/search", json={"query": "zoom", "top_k": 50})
    contents = [result["content"] for result in response.json()["results"]
                if result["source_path"] == "map/base.js"]
    assert contents
    assert all("zoom" in content for content in contents)


# --- modify two files: exactly those two are reindexed ----------------------

@requires_qdrant
def test_modifying_two_files_reindexes_exactly_those_two(corpus):
    assert ingestion_main.main() == 0
    before = _points_by_file()
    changed_texts = {
        "kdk/map/base.js": "export function pan () {\n  return [1, 1]\n}\n",
        "kdk/docs/guide.md": "# Guide\n\n## Update\n\nRun yarn upgrade.\n",
    }
    for source_path, text in changed_texts.items():
        _write(corpus.root, source_path, text)

    assert ingestion_main.main() == 0

    after = _points_by_file()
    for source_path, text in changed_texts.items():
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        assert ({point.payload["file_sha1"]
                 for point in after[_key(source_path)]} == {digest})
    untouched = set(before) - {_key(path) for path in changed_texts}
    for key in untouched:
        assert ({point.id for point in after[key]}
                == {point.id for point in before[key]})
    assert corpus.embed_calls[-1] == sum(
        len(after[_key(path)]) for path in changed_texts)


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------


# (repository, repo-relative path) key of a workspace-relative path.
def _key(source_path):
    repository, relative = source_path.split("/", 1)
    return repository, relative


# Every stored point, grouped by (repository, source_path).
def _points_by_file():
    client = vectordb._get_qdrant_client()
    records, next_page = client.scroll(
        collection_name=CODE_COLLECTION, limit=1000, with_payload=True,
        with_vectors=False)
    # A next page would mean the assertions only see part of the collection.
    assert next_page is None, "test corpus outgrew the one-shot scroll"
    grouped = {}
    for record in records:
        key = (record.payload["repository"], record.payload["source_path"])
        grouped.setdefault(key, []).append(record)
    return grouped
