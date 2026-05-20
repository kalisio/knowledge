from fastapi.testclient import TestClient

import api.app as api_app
from api.app import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_endpoint_accepts_post_json(monkeypatch) -> None:
    def fake_answer_question(question: str):
        assert question == "How does KDK define map layers?"
        return {
            "answer": "A test answer.",
            "sources": [],
            "provider": "test",
            "model": "test-model",
        }

    monkeypatch.setattr(api_app, "answer_question", fake_answer_question)
    client = TestClient(app)
    response = client.post("/ask", json={"question": "How does KDK define map layers?"})
    assert response.status_code == 200
    assert response.json() == {
        "answer": "A test answer.",
        "sources": [],
        "provider": "test",
        "model": "test-model",
    }


def test_ask_endpoint_rejects_get() -> None:
    client = TestClient(app)
    response = client.get("/ask?question=How does KDK define map layers?")
    assert response.status_code == 405


def test_search_endpoint_returns_chunks_without_llm(monkeypatch) -> None:
    captured = {}

    def fake_search(query: str, top_k: int):
        captured["query"] = query
        captured["top_k"] = top_k
        from api.schemas import SearchResponse, SearchResultChunk
        return SearchResponse(
            results=[
                SearchResultChunk(
                    path="kano/src/store/layers.js",
                    repo="kano",
                    lines="45-78",
                    score=0.92,
                    content="export const layerStore = ...",
                    commit_history=["feat: add layer store", "fix: layer ordering"],
                ),
            ]
        )

    monkeypatch.setattr(api_app, "search_chunks", fake_search)
    client = TestClient(app)
    response = client.post("/search", json={"query": "layer store", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert captured == {"query": "layer store", "top_k": 3}
    assert body == {
        "results": [
            {
                "path": "kano/src/store/layers.js",
                "repo": "kano",
                "lines": "45-78",
                "score": 0.92,
                "content": "export const layerStore = ...",
                "commit_history": ["feat: add layer store", "fix: layer ordering"],
            }
        ]
    }


def test_search_endpoint_top_k_defaults_to_5() -> None:
    captured = {}

    def fake_search(query: str, top_k: int):
        captured["top_k"] = top_k
        from api.schemas import SearchResponse
        return SearchResponse(results=[])

    client = TestClient(app)
    import api.app as _app_mod
    _app_mod.search_chunks = fake_search
    response = client.post("/search", json={"query": "anything"})
    assert response.status_code == 200
    assert captured["top_k"] == 5


def test_search_endpoint_rejects_top_k_out_of_range() -> None:
    client = TestClient(app)
    response = client.post("/search", json={"query": "x", "top_k": 0})
    assert response.status_code == 422
    response = client.post("/search", json={"query": "x", "top_k": 99})
    assert response.status_code == 422
