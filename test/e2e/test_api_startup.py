"""Starting the API: boot-time checks, configuration, auth and error codes.

The first leg of the journey -- before anything is ingested, the service has
to come up, say what it is configured with, and answer sanely when the index
is empty or the vector database is down.
"""

import time

import jwt
import pytest
from fastapi.testclient import TestClient

import ingestion.clients.vectordb as vectordb
from api.main import app

from conftest import base_env
from helpers import CODE_COLLECTION, VECTOR_SIZE, requires_qdrant

SECRET = "test-secret-please-use-32-plus-bytes"


# Mint a token the way utils/make-jwt.py does.
def make_token(secret=SECRET, aud="kalisio", iss="kalisio", ttl=3600,
               algorithm="HS256"):
    now = int(time.time())
    return jwt.encode(
        {"sub": "test", "aud": aud, "iss": iss, "iat": now, "exp": now + ttl},
        secret, algorithm=algorithm)


@pytest.mark.Startup
class TestStartup:
    # The service comes up and reports its effective configuration.
    @requires_qdrant
    def test_the_api_starts_and_logs_its_configuration(self, pipeline):
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200

        logs = pipeline.logs.text
        assert "starting knowledge API" in logs
        assert CODE_COLLECTION in logs
        assert "embedding model=test-model" in logs
        assert "llm model=test-llm" in logs

    # Secrets are reported as present/absent, never printed.
    @requires_qdrant
    def test_the_startup_banner_never_prints_a_secret(self, pipeline):
        with TestClient(app):
            pass

        logs = pipeline.logs.text
        assert "LLM_API_KEY=set" in logs
        assert "test-key" not in logs

    # An index that was never built is called out by name, with the command
    # that fills it -- the operator must not have to guess.
    @requires_qdrant
    def test_an_empty_index_is_reported_at_startup(self, pipeline):
        vectordb.ensure_collection(CODE_COLLECTION, VECTOR_SIZE)

        with TestClient(app):
            pass

        logs = pipeline.logs.text
        assert "EMPTY" in logs
        assert "python -m ingestion.bin" in logs

    # Once ingestion has run, the same banner reports how much is indexed.
    @requires_qdrant
    def test_a_populated_index_is_reported_at_startup(self, pipeline):
        pipeline.workspace.install_samples()
        assert pipeline.run() == 0
        pipeline.logs.clear()

        with TestClient(app):
            pass

        assert "indexed chunks" in pipeline.logs.text
        assert "EMPTY" not in pipeline.logs.text

    # /health is the liveness probe: it must answer without a token, and it
    # must not depend on Qdrant being up.
    @requires_qdrant
    def test_health_needs_no_token(self, pipeline):
        response = pipeline.client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    # A missing required setting fails fast, naming the variable at fault.
    def test_a_missing_required_setting_names_the_variable(self, configure):
        configure(**{**base_env("/tmp/nowhere", "http://localhost:6333"),
                     "QDRANT_URL": None})

        with pytest.raises(RuntimeError, match="QDRANT_URL"):
            with TestClient(app):
                pass

    # Auth on without a secret cannot verify any token: refuse to start
    # rather than 500 on every request.
    def test_auth_without_a_secret_refuses_to_start(self, configure):
        configure(**{**base_env("/tmp/nowhere", "http://localhost:6333"),
                     "KNOWLEDGE_AUTH_ENABLED": "true", "APP_SECRET": None})

        with pytest.raises(RuntimeError, match="APP_SECRET"):
            with TestClient(app):
                pass

    # A vector database that is down must not keep the service from starting:
    # /health is what tells an orchestrator the process is alive, and the
    # lifespan banner is best-effort by design. An unreachable Qdrant used to
    # exit the process from inside the banner, crash-looping the service.
    def test_the_api_starts_even_when_qdrant_is_down(self, configure):
        configure(**base_env("/tmp/nowhere", "http://127.0.0.1:1/"))

        with TestClient(app) as client:
            assert client.get("/health").status_code == 200


@pytest.mark.Startup
class TestAuth:
    # Turn auth on with a known secret.
    @pytest.fixture
    def auth_on(self, pipeline, configure):
        configure(**{**base_env(pipeline.development_dir, pipeline.qdrant_url),
                     "KNOWLEDGE_AUTH_ENABLED": "true", "APP_SECRET": SECRET})
        return pipeline

    # No token at all.
    def test_a_request_without_a_token_is_rejected(self, auth_on):
        response = auth_on.client.post("/ask", json={"question": "hi"})

        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"].startswith("Bearer")

    # Every way a token can be wrong ends in the same 401, never a 500.
    @pytest.mark.parametrize("token,label", [
        ("not-a-jwt-at-all", "garbage"),
        (make_token(secret="another-secret-entirely-32-bytes"), "wrong key"),
        (make_token(aud="somebody-else"), "wrong audience"),
        (make_token(iss="somebody-else"), "wrong issuer"),
        (make_token(ttl=-3600), "expired"),
    ])
    def test_a_bad_token_is_rejected(self, auth_on, token, label):
        response = auth_on.client.post(
            "/ask", json={"question": "hi"},
            headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 401, label

    # An unsigned token must never be accepted: the decoder pins the
    # algorithm, so "alg: none" cannot talk its way in.
    def test_an_unsigned_token_is_rejected(self, auth_on):
        unsigned = jwt.encode(
            {"sub": "attacker", "aud": "kalisio", "iss": "kalisio"},
            key="", algorithm="none")

        response = auth_on.client.post(
            "/ask", json={"question": "hi"},
            headers={"Authorization": f"Bearer {unsigned}"})

        assert response.status_code == 401

    # A token in the wrong scheme is not a bearer token.
    def test_a_non_bearer_scheme_is_rejected(self, auth_on):
        response = auth_on.client.post(
            "/ask", json={"question": "hi"},
            headers={"Authorization": f"Basic {make_token()}"})

        assert response.status_code == 401


@pytest.mark.Startup
class TestRequestValidation:
    # The request contract is enforced before anything is embedded.
    @pytest.mark.parametrize("payload", [
        {},                                  # question missing
        {"question": ""},                    # below min_length
        {"question": "x" * 2001},            # above max_length
    ])
    def test_an_invalid_ask_payload_is_rejected(self, pipeline, payload):
        response = pipeline.client.post("/ask", json=payload)

        assert response.status_code == 422

    @pytest.mark.parametrize("payload", [
        {"top_k": 5},                        # query missing
        {"query": "x", "top_k": 0},          # below ge=1
        {"query": "x", "top_k": 51},         # above le=50
        {"query": "x", "top_k": "many"},     # not an int
    ])
    def test_an_invalid_search_payload_is_rejected(self, pipeline, payload):
        response = pipeline.client.post("/search", json=payload)

        assert response.status_code == 422

    # Querying before ingestion has run is a 503 naming the command to run,
    # not an empty 200 that looks like "nothing matches".
    @requires_qdrant
    def test_a_query_before_any_ingestion_returns_503(self, pipeline):
        response = pipeline.client.post(
            "/search", json={"query": "layer", "top_k": 5})

        assert response.status_code == 503
        assert "python -m ingestion.bin" in response.json()["detail"]
        assert "empty index" in pipeline.logs.text
