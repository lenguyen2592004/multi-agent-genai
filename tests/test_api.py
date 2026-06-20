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
            "document_id": "sample_policy",
            "query": "Summarize this document and extract action items",
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "trace_id" in body
    assert "answer" in body
    assert "trace" in body


def test_query_endpoint_accepts_document_filter() -> None:
    client = TestClient(app)
    response = client.post(
        "/query",
        json={
            "user_id": "api-test",
            "document_id": "sample_policy",
            "query": "What action items are in the policy?",
            "top_k": 2,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "trace_id" in body
