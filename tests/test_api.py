from fastapi.testclient import TestClient

import api
from supportflow.model import Prediction


class FakeModel:
    def predict(self, text: str, similar_count: int = 3) -> Prediction:
        return Prediction(
            intent="pending_transfer",
            intent_display="Pending transfer",
            team="Transfers & Beneficiaries",
            priority="Medium",
            priority_reason="Rule-based test priority.",
            sla_hours=8,
            sla_reason="Demo policy: first response is due within 8 clock hours.",
            confidence=0.9,
            decision="Automatically route",
            threshold=0.55,
            top_predictions=[],
            similar_tickets=[{"message": "Example", "intent": "Pending transfer", "similarity": 0.8}][
                :similar_count
            ],
            inference_ms=2.0,
        )


def client(monkeypatch) -> TestClient:
    monkeypatch.setattr(api, "get_model", lambda: FakeModel())
    return TestClient(api.app)


def test_service_and_health_endpoints() -> None:
    test_client = TestClient(api.app)
    assert test_client.get("/").json()["interactive_docs"] == "/docs"
    health = test_client.get("/health")
    assert health.status_code == 200
    assert health.json()["artifacts_ready"] is True


def test_predict_returns_routing_payload(monkeypatch) -> None:
    response = client(monkeypatch).post(
        "/v1/predict",
        json={"message": "Where is my transfer?", "similar_count": 1},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "pending_transfer"
    assert payload["priority"] == "Medium"
    assert payload["sla_hours"] == 8
    assert len(payload["similar_tickets"]) == 1


def test_predict_rejects_invalid_ticket_text(monkeypatch) -> None:
    response = client(monkeypatch).post("/v1/predict", json={"message": "   "})
    assert response.status_code == 422


def test_batch_predicts_multiple_tickets(monkeypatch) -> None:
    response = client(monkeypatch).post(
        "/v1/predict/batch",
        json={"messages": ["Where is my transfer?", "My transfer is still pending"]},
    )
    assert response.status_code == 200
    assert response.json()["count"] == 2


def test_batch_limit_is_enforced(monkeypatch) -> None:
    response = client(monkeypatch).post(
        "/v1/predict/batch",
        json={"messages": ["A valid customer support message"] * 51},
    )
    assert response.status_code == 422
