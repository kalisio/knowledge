"""The auth dependency on its own: who gets through and who does not."""

import time
from types import SimpleNamespace

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

import api.services.security as auth

SECRET = "test-secret-please-use-32-plus-bytes"


def config(enabled=True, secret=SECRET):
    return SimpleNamespace(
        auth_enabled=enabled, app_secret=secret, jwt_algorithm="HS256",
        jwt_audience="kalisio", jwt_issuer="kalisio")


def token(secret=SECRET, aud="kalisio", iss="kalisio", ttl=3600, sub="test"):
    now = int(time.time())
    return jwt.encode({"sub": sub, "aud": aud, "iss": iss, "iat": now,
                       "exp": now + ttl}, secret, algorithm="HS256")


def bearer(value):
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=value)


def test_a_valid_token_returns_its_claims(monkeypatch):
    monkeypatch.setattr(auth, "get_config", config)

    claims = auth.verify_jwt(bearer(token(sub="tristan")))

    assert claims["sub"] == "tristan"


def test_auth_disabled_lets_everything_through(monkeypatch):
    monkeypatch.setattr(auth, "get_config", lambda: config(enabled=False))

    assert auth.verify_jwt(None) is None


def test_a_missing_token_is_a_401(monkeypatch):
    monkeypatch.setattr(auth, "get_config", config)

    with pytest.raises(HTTPException) as raised:
        auth.verify_jwt(None)

    assert raised.value.status_code == 401
    assert raised.value.detail == "missing_bearer_token"


def test_a_non_bearer_scheme_is_a_401(monkeypatch):
    monkeypatch.setattr(auth, "get_config", config)
    credentials = HTTPAuthorizationCredentials(
        scheme="Basic", credentials=token())

    with pytest.raises(HTTPException) as raised:
        auth.verify_jwt(credentials)

    assert raised.value.status_code == 401


def test_auth_on_without_a_secret_fails_closed(monkeypatch):
    # Refusing is the only safe answer: with no secret nothing can be
    # verified, and running open would let every token through.
    monkeypatch.setattr(auth, "get_config", lambda: config(secret=None))

    with pytest.raises(HTTPException) as raised:
        auth.verify_jwt(bearer(token()))

    assert raised.value.status_code == 500
    assert "APP_SECRET" in raised.value.detail


@pytest.mark.parametrize("bad, label", [
    (token(secret="another-secret-entirely-32-bytes"), "wrong key"),
    (token(aud="somebody-else"), "wrong audience"),
    (token(iss="somebody-else"), "wrong issuer"),
    (token(ttl=-3600), "expired"),
    ("not-a-jwt", "garbage"),
])
def test_an_invalid_token_is_a_401(monkeypatch, bad, label):
    monkeypatch.setattr(auth, "get_config", config)

    with pytest.raises(HTTPException) as raised:
        auth.verify_jwt(bearer(bad))

    assert raised.value.status_code == 401, label
    assert raised.value.headers["WWW-Authenticate"].startswith("Bearer")
