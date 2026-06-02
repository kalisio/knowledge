"""API tests: /health is open; /ask and /search require a valid JWT."""

import time

import jwt
import pytest
from fastapi.testclient import TestClient

import api.auth as auth
from api.app import app


client = TestClient(app)

# HS256 keys should be >= 32 bytes (see RFC 7518); keep the test realistic.
TEST_SECRET = "test-secret-please-use-32-plus-bytes"


# Mint a token the way scripts/make-jwt.py does, signed with TEST_SECRET.
def make_token(secret=TEST_SECRET, aud="kalisio", iss="kalisio", ttl=3600):
    now = int(time.time())
    payload = {
        "sub": "test", "aud": aud, "iss": iss, "iat": now, "exp": now + ttl,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


# Turn auth on with a known secret, only for the test that asks for it.
@pytest.fixture
def auth_on(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_ENABLED", True)
    monkeypatch.setattr(auth, "APP_SECRET", TEST_SECRET)


def test_health_is_open():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_without_token_is_rejected(auth_on):
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 401


def test_ask_with_garbage_token_is_rejected(auth_on):
    headers = {"Authorization": "Bearer not-a-real-token"}
    response = client.post("/ask", json={"question": "hi"}, headers=headers)
    assert response.status_code == 401


def test_ask_with_wrong_audience_is_rejected(auth_on):
    headers = {"Authorization": f"Bearer {make_token(aud='someone-else')}"}
    response = client.post("/ask", json={"question": "hi"}, headers=headers)
    assert response.status_code == 401


def test_ask_with_valid_token_is_accepted(auth_on):
    headers = {"Authorization": f"Bearer {make_token()}"}
    response = client.post("/ask", json={"question": "hi"}, headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "sources" in body


def test_search_with_valid_token_is_accepted(auth_on):
    headers = {"Authorization": f"Bearer {make_token()}"}
    response = client.post(
        "/search", json={"query": "hi", "top_k": 3}, headers=headers,
    )
    assert response.status_code == 200
    assert "results" in response.json()


def test_auth_disabled_allows_no_token(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_ENABLED", False)
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 200
