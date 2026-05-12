from fastapi.testclient import TestClient

import knowledge.src.app as api_app
from knowledge.src.app import app


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
