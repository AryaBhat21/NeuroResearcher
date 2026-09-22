"""
End-to-end integration tests for FastAPI endpoints.
"""
from fastapi.testclient import TestClient


def test_create_and_list_conversations(client: TestClient):
    """Verifies conversation lifecycle endpoints."""
    # Create conversation
    create_res = client.post("/conversations", json={"title": "ALS Biomarker Review"})
    assert create_res.status_code == 201
    data = create_res.json()
    assert data["title"] == "ALS Biomarker Review"
    conv_id = data["id"]

    # List conversations
    list_res = client.get("/conversations")
    assert list_res.status_code == 200
    conv_list = list_res.json()
    assert any(c["id"] == conv_id for c in conv_list)

    # Get single conversation
    detail_res = client.get(f"/conversations/{conv_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["id"] == conv_id

    # Delete conversation
    del_res = client.delete(f"/conversations/{conv_id}")
    assert del_res.status_code == 204

    # Verify not found after delete
    get_after = client.get(f"/conversations/{conv_id}")
    assert get_after.status_code == 404


def test_chat_api_turn_execution(client: TestClient):
    """Verifies end-to-end /chat endpoint with tool invocation."""
    response = client.post(
        "/chat",
        json={"message": "Search recent literature on glioblastoma immunotherapy"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "conversation_id" in data
    assert "response" in data
    assert "tools_used" in data
    assert len(data["response"]) > 0


def test_research_queries_endpoints(client: TestClient):
    """Verifies Phase 1 & 2 research-queries endpoints."""
    # POST query
    create_res = client.post("/research-queries", json={"query": "Lewy body dementia pathology"})
    assert create_res.status_code == 201
    q_id = create_res.json()["id"]

    # GET all
    get_all_res = client.get("/research-queries")
    assert get_all_res.status_code == 200
    assert any(q["id"] == q_id for q in get_all_res.json())

    # GET by ID
    get_one_res = client.get(f"/research-queries/{q_id}")
    assert get_one_res.status_code == 200
    assert get_one_res.json()["query"] == "Lewy body dementia pathology"


def test_chat_api_validation_error(client: TestClient):
    """Verifies that empty prompt triggers 422 Unprocessable Entity."""
    res = client.post("/chat", json={"message": ""})
    assert res.status_code == 422
