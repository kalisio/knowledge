"""API tests: /health is open; /ask and /search require a valid JWT."""

import os
import time
import urllib.request
from types import SimpleNamespace

import jwt
import pytest
from fastapi.testclient import TestClient

import api.app as app_module
import api.auth as auth
import api.handlers as handlers
import utils.vectordb as vectordb
from api.app import app


client = TestClient(app)

TEST_SECRET = "test-secret-please-use-32-plus-bytes"
TEST_COLLECTION = "knowledge_pytest"


# Mint a token the way scripts/make-jwt.py does, signed with TEST_SECRET.
def make_token(secret=TEST_SECRET, aud="kalisio", iss="kalisio", ttl=3600):
    now = int(time.time())
    payload = {
        "sub": "test", "aud": aud, "iss": iss, "iat": now, "exp": now + ttl,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


# A stand-in for ApiConfig holding only the auth fields verify_jwt reads, so
# the auth tests don't need the full (LLM/Qdrant) env to be configured.
def fake_config(enabled=True, secret=TEST_SECRET):
    return SimpleNamespace(
        auth_enabled=enabled, app_secret=secret,
        jwt_algorithm="HS256", jwt_audience="kalisio", jwt_issuer="kalisio")


# True when Qdrant answers at QDRANT_URL. The /ask and /search tests query
# Qdrant through the handler, so they are skipped when it is unreachable.
def qdrant_reachable():
    url = os.environ.get("QDRANT_URL")
    if not url:
        return False
    try:
        urllib.request.urlopen(f"{url.rstrip('/')}/healthz", timeout=2)
        return True
    except Exception:
        return False


requires_qdrant = pytest.mark.skipif(
    not qdrant_reachable(), reason="needs a running Qdrant (QDRANT_URL)")


@pytest.mark.General
class TestApi:
    # Turn auth on with a known secret, only for the tests that ask for it.
    @pytest.fixture
    def auth_on(self, monkeypatch):
        monkeypatch.setattr(auth, "get_config", lambda: fake_config())

    # Seed a throwaway collection so /ask and /search run their upsert and
    # search against Qdrant; the embedding model and the LLM are stubbed.
    @pytest.fixture
    def seeded_qdrant(self, monkeypatch):
        cfg = SimpleNamespace(
            qdrant_url=os.environ["QDRANT_URL"],
            qdrant_collection_code=TEST_COLLECTION,
            embedding_model="test", embedding_batch_size=8)
        monkeypatch.setattr(vectordb, "get_runtime_config", lambda: cfg)
        vector = [1.0, 0.0, 0.0, 0.0]
        chunk = {"text": "export const base = []", "metadata": {
            "source_path": "kano/store/layers.js", "repository": "kano",
            "breadcrumb": "layers > base", "chunk_index": 0,
            "file_sha1": "test"}}
        vectordb.ensure_collection(len(vector), recreate=True)
        vectordb.upsert([chunk], [vector])
        monkeypatch.setattr(handlers, "get_config", lambda: SimpleNamespace(
            top_k=5, max_context_chars=14000,
            prompt_template="Context:\n{context}\n\nQuestion: {question}"))
        monkeypatch.setattr(handlers.embeddings, "encode", lambda text: vector)
        monkeypatch.setattr(handlers.llm, "ask", lambda prompt: SimpleNamespace(
            answer="stub answer", provider="stub", model="stub-model"))
        yield
        vectordb._get_client().delete_collection(TEST_COLLECTION)

    # /health needs no auth and returns ok.
    def test_health_is_open(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    # /ask without a token is rejected with 401.
    def test_ask_without_token_is_rejected(self, auth_on):
        response = client.post("/ask", json={"question": "hi"})
        assert response.status_code == 401

    # /ask with a malformed token is rejected with 401.
    def test_ask_with_garbage_token_is_rejected(self, auth_on):
        headers = {"Authorization": "Bearer not-a-real-token"}
        response = client.post(
            "/ask", json={"question": "hi"}, headers=headers)
        assert response.status_code == 401

    # /ask with a token for the wrong audience is rejected with 401.
    def test_ask_with_wrong_audience_is_rejected(self, auth_on):
        headers = {"Authorization": f"Bearer {make_token(aud='someone-else')}"}
        response = client.post(
            "/ask", json={"question": "hi"}, headers=headers)
        assert response.status_code == 401

    # /ask with a valid token returns an answer and its sources.
    @requires_qdrant
    def test_ask_with_valid_token_is_accepted(self, auth_on, seeded_qdrant):
        headers = {"Authorization": f"Bearer {make_token()}"}
        response = client.post(
            "/ask", json={"question": "hi"}, headers=headers)
        assert response.status_code == 200
        body = response.json()
        assert "answer" in body
        assert "sources" in body

    # /search with a valid token returns results.
    @requires_qdrant
    def test_search_with_valid_token_is_accepted(self, auth_on, seeded_qdrant):
        headers = {"Authorization": f"Bearer {make_token()}"}
        response = client.post(
            "/search", json={"query": "hi", "top_k": 3}, headers=headers,
        )
        assert response.status_code == 200
        assert "results" in response.json()

    # When auth is disabled, /ask works without any token.
    @requires_qdrant
    def test_auth_disabled_allows_no_token(self, monkeypatch, seeded_qdrant):
        monkeypatch.setattr(
            auth, "get_config", lambda: fake_config(enabled=False))
        response = client.post("/ask", json={"question": "hi"})
        assert response.status_code == 200

    # With auth ON but APP_SECRET missing, the service refuses to start: the
    # lifespan raises, so entering the TestClient (startup) errors out instead
    # of booting into a state where every authenticated request 500s.
    def test_startup_fails_when_auth_on_without_secret(self, monkeypatch):
        broken = SimpleNamespace(
            log_level="INFO", auth_enabled=True, app_secret=None)
        monkeypatch.setattr(app_module, "get_config", lambda: broken)
        with pytest.raises(RuntimeError, match="APP_SECRET"):
            with TestClient(app):
                pass
