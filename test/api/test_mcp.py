"""The MCP server: the tool contract, the token in front of it, the mount."""

import time

import jwt
import pytest
from starlette.testclient import TestClient

import api.main as main
import api.services.retrieval as retrieval
from api.services.mcp import SEARCH_CODE_DESCRIPTION

SECRET = "test-secret-please-use-32-plus-bytes"

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}

CHUNKS = [{
    "path": "src/store/layers.js", "lines": "45-78", "score": 0.92,
    "content": "export const layers = {}",
    "commit_history": ["feat: add layers"], "repo": "kano",
    "breadcrumb": "store > layers", "chunk_index": 0,
}]


def token(ttl=3600, sub="test"):
    now = int(time.time())
    return jwt.encode({"sub": sub, "aud": "kalisio", "iss": "kalisio",
                       "iat": now, "exp": now + ttl},
                      SECRET, algorithm="HS256")


# One authenticated client per test, over a fresh app. The TestClient
# context runs the lifespan, which the MCP session manager needs.
@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("APP_SECRET", SECRET)
    monkeypatch.setattr(main.vectordb, "count_chunks", lambda: len(CHUNKS))
    with TestClient(main.create_app()) as testclient:
        yield testclient


def rpc(client, payload, authenticated=True):
    headers = dict(HEADERS)
    if authenticated:
        headers["Authorization"] = f"Bearer {token()}"
    return client.post("/mcp", json=payload, headers=headers)


def initialize(client, authenticated=True):
    return rpc(client, {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "test", "version": "0"}},
    }, authenticated)


def list_tools(client):
    response = rpc(client, {"jsonrpc": "2.0", "id": 2,
                            "method": "tools/list"})
    return response.json()["result"]["tools"]


def call_tool(client, name, arguments):
    return rpc(client, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                        "params": {"name": name, "arguments": arguments}})


def test_the_tool_list_is_the_contract(client):
    tools = list_tools(client)

    # One tool exactly: adding or dropping one must be a conscious change.
    assert [tool["name"] for tool in tools] == ["search_code"]
    assert tools[0]["description"] == SEARCH_CODE_DESCRIPTION
    properties = tools[0]["inputSchema"]["properties"]
    assert set(properties) == {"query", "top_k"}
    assert properties["top_k"]["default"] == 5

    # The same bounds POST /search enforces, so one transport cannot ask
    # for what the other refuses.
    assert properties["top_k"]["minimum"] == 1
    assert properties["top_k"]["maximum"] == 50
    assert properties["query"]["maxLength"] == 2000


def test_a_call_passes_through_to_the_retrieval_service(client, monkeypatch):
    calls = []

    def search_chunks(query, top_k):
        calls.append((query, top_k))
        return CHUNKS

    monkeypatch.setattr(retrieval, "search_chunks", search_chunks)

    response = call_tool(client, "search_code",
                         {"query": "layer permissions", "top_k": 3})

    assert response.status_code == 200
    assert calls == [("layer permissions", 3)]
    result = response.json()["result"]
    assert result["isError"] is False
    assert result["structuredContent"]["result"] == CHUNKS


def test_top_k_defaults_to_five(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        retrieval, "search_chunks",
        lambda query, top_k: calls.append((query, top_k)) or CHUNKS)

    call_tool(client, "search_code", {"query": "geolocation"})

    assert calls == [("geolocation", 5)]


def test_a_retrieval_error_is_a_tool_error_not_a_crash(client, monkeypatch):
    def search_chunks(query, top_k):
        raise RuntimeError("qdrant is down")

    monkeypatch.setattr(retrieval, "search_chunks", search_chunks)

    response = call_tool(client, "search_code", {"query": "anything"})

    # The protocol answer stays 200; the failure travels as isError, which
    # the agent can read and react to.
    assert response.status_code == 200
    assert response.json()["result"]["isError"] is True


def test_no_token_is_a_401(client):
    response = initialize(client, authenticated=False)

    assert response.status_code == 401


def test_an_expired_token_is_a_401(client):
    headers = {**HEADERS, "Authorization": f"Bearer {token(ttl=-3600)}"}

    response = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1,
                                         "method": "tools/list"},
                           headers=headers)

    assert response.status_code == 401


def test_a_header_that_is_not_utf8_is_a_401(client):
    # Header bytes are not guaranteed to decode as UTF-8. Reading them as
    # one turned a malformed token into a 500 through the auth path.
    headers = [(b"content-type", b"application/json"),
               (b"accept", b"application/json, text/event-stream"),
               (b"authorization", b"Bearer \xff")]

    response = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1,
                                         "method": "tools/list"},
                           headers=headers)

    assert response.status_code == 401
    # invalid_token, not missing_bearer_token: the header did reach the
    # middleware, it just does not carry a usable one.
    assert response.json()["detail"] == "invalid_token"


def test_auth_disabled_lets_the_call_through(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_AUTH_ENABLED", "false")
    monkeypatch.delenv("APP_SECRET", raising=False)
    monkeypatch.setattr(main.vectordb, "count_chunks", lambda: len(CHUNKS))
    monkeypatch.setattr(retrieval, "search_chunks",
                        lambda query, top_k: CHUNKS)

    with TestClient(main.create_app()) as client:
        response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers=HEADERS)

    assert response.status_code == 200


def test_an_out_of_range_top_k_never_reaches_retrieval(client, monkeypatch):
    calls = []
    monkeypatch.setattr(retrieval, "search_chunks",
                        lambda query, top_k: calls.append((query, top_k)))

    response = call_tool(client, "search_code",
                         {"query": "layers", "top_k": 9999})

    assert response.json()["result"]["isError"] is True
    assert calls == []


def test_one_app_can_be_started_twice(monkeypatch):
    # A session manager runs once, so each startup gets its own sub-app.
    monkeypatch.setenv("APP_SECRET", SECRET)
    monkeypatch.setattr(main.vectordb, "count_chunks", lambda: len(CHUNKS))
    app = main.create_app()

    for _ in range(2):
        with TestClient(app) as started:
            assert started.get("/health").status_code == 200


def test_both_spellings_of_the_endpoint_answer_without_a_redirect(client):
    # /mcp used to 307 onto /mcp/, which is what broke the deployed server:
    # the Location carried the scheme uvicorn sees (http), nginx bounced it
    # back to /mcp without the slash, and the client looped until it failed
    # with TooManyRedirects. Neither spelling may redirect.
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    headers = {**HEADERS, "Authorization": f"Bearer {token()}"}

    for path in ("/mcp", "/mcp/"):
        response = client.post(path, json=payload, headers=headers,
                               follow_redirects=False)

        assert response.status_code == 200, path
        assert response.json()["result"]["tools"][0]["name"] == "search_code"


def test_an_unauthenticated_call_is_refused_before_any_redirect(client):
    # The token is checked in front of the mount, so a missing one is a 401
    # on both spellings -- never a redirect that leaks the path elsewhere.
    for path in ("/mcp", "/mcp/"):
        response = client.post(path, json={"jsonrpc": "2.0", "id": 1,
                                           "method": "tools/list"},
                               headers=HEADERS, follow_redirects=False)

        assert response.status_code == 401, path


def test_the_mount_leaves_the_rest_of_the_api_alone(client):
    # Mounted under /mcp, not at the root: an unknown path is still a 404
    # and a wrong method still a 405, both in the JSON the API answers with.
    assert client.get("/health").status_code == 200
    assert client.get("/nowhere").status_code == 404
    assert client.get("/nowhere").json() == {"detail": "Not Found"}
    assert client.get("/search").status_code == 405
