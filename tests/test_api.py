"""API-level tests for Space ping and reset behavior."""

from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_space_url_ping_returns_200() -> None:
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload.get("status") == "ok"
    assert "/reset" in payload.get("endpoints", [])


def test_reset_responds_with_observation() -> None:
    response = client.post("/reset", json={"task_id": "easy"})
    assert response.status_code == 200
    payload = response.json()
    assert payload.get("done") is False
    assert payload.get("reward") is None

    observation = payload.get("observation", {})
    assert isinstance(observation.get("servers"), list)
    assert len(observation["servers"]) > 0
