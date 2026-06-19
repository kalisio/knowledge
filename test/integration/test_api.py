"""API tests: /health is open; /ask and /search require a valid JWT."""

import time
from types import SimpleNamespace

import jwt
import pytest
from fastapi.testclient import TestClient

import api.app as app_module
import api.auth as auth
from api.app import app


client = TestClient(app)

TEST_SECRET = "test-secret-please-use-32-plus-bytes"


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


# Turn auth on with a known secret, only for the test that asks for it.
@pytest.fixture
def auth_on(monkeypatch):
    monkeypatch.setattr(auth, "get_config", lambda: fake_config())


# /health needs no auth and returns ok.
def test_health_is_open():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# /ask without a token is rejected with 401.
def test_ask_without_token_is_rejected(auth_on):
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 401


# /ask with a malformed token is rejected with 401.
def test_ask_with_garbage_token_is_rejected(auth_on):
    headers = {"Authorization": "Bearer not-a-real-token"}
    response = client.post("/ask", json={"question": "hi"}, headers=headers)
    assert response.status_code == 401


# /ask with a token for the wrong audience is rejected with 401.
def test_ask_with_wrong_audience_is_rejected(auth_on):
    headers = {"Authorization": f"Bearer {make_token(aud='someone-else')}"}
    response = client.post("/ask", json={"question": "hi"}, headers=headers)
    assert response.status_code == 401


# /ask with a valid token returns an answer and its sources.
def test_ask_with_valid_token_is_accepted(auth_on):
    headers = {"Authorization": f"Bearer {make_token()}"}
    response = client.post("/ask", json={"question": "hi"}, headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "sources" in body


# /search with a valid token returns results.
def test_search_with_valid_token_is_accepted(auth_on):
    headers = {"Authorization": f"Bearer {make_token()}"}
    response = client.post(
        "/search", json={"query": "hi", "top_k": 3}, headers=headers,
    )
    assert response.status_code == 200
    assert "results" in response.json()


# When auth is disabled, /ask works without any token.
def test_auth_disabled_allows_no_token(monkeypatch):
    monkeypatch.setattr(auth, "get_config", lambda: fake_config(enabled=False))
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 200


# With auth ON but APP_SECRET missing, the service refuses to start: the
# lifespan raises, so entering the TestClient (which runs startup) errors out
# instead of booting into a state where every authenticated request 500s.
def test_startup_fails_when_auth_on_without_secret(monkeypatch):
    broken = SimpleNamespace(
        log_level="INFO", auth_enabled=True, app_secret=None)
    monkeypatch.setattr(app_module, "get_config", lambda: broken)
    with pytest.raises(RuntimeError, match="APP_SECRET"):
        with TestClient(app):
            pass
