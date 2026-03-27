from fastapi.testclient import TestClient

from api.main import app


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_endpoint() -> None:
    client = TestClient(app)
    response = client.post(
        "/query",
        json={
            "user_id": "api-test",
            "query": "Summarize this document and extract action items",
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "trace_id" in body
    assert "answer" in body
    assert "trace" in body
