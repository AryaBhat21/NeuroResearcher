"""
Tests for health check and system diagnostic endpoints.
"""
from fastapi.testclient import TestClient


def test_health_check_status_code(client: TestClient):
    """Verifies that the /health endpoint returns HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_check_payload_structure(client: TestClient):
    """Verifies that the /health endpoint returns expected schema fields."""
    response = client.get("/health")
    data = response.json()

    assert "status" in data
    assert "database" in data
    assert "agent_mode" in data
    assert "timestamp" in data
    assert data["status"] in ("healthy", "degraded")
