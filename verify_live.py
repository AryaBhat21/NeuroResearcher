"""
Live end-to-end verification script for NeuroResearch.
Tests health check, conversation creation, multi-turn chat with coreference,
and PostgreSQL persistence.
"""
from fastapi.testclient import TestClient
from main import app
from database import SessionLocal, Conversation, Message, check_db_connection


def main():
    print("=" * 65)
    print(" NeuroResearch — Live End-to-End Verification")
    print("=" * 65)

    client = TestClient(app)

    # 1. Health & Database Connectivity
    print("\n[Step 1] Verifying /health & PostgreSQL Connectivity...")
    health_resp = client.get("/health")
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
    health_data = health_resp.json()
    print(f" -> Status: {health_data['status']}")
    print(f" -> Database: {health_data['database']}")
    print(f" -> Agent Mode: {health_data['agent_mode']}")

    # 2. Create Conversation
    print("\n[Step 2] Creating New Research Conversation...")
    conv_resp = client.post("/conversations", json={"title": "Neurodegenerative Oculomotor Biomarkers"})
    assert conv_resp.status_code == 201, f"Failed to create conversation: {conv_resp.text}"
    conv_data = conv_resp.json()
    cid = conv_data["id"]
    print(f" -> Created Conversation ID: {cid} (Title: '{conv_data['title']}')")

    # 3. Send First Message (Requires Tool Execution)
    print("\n[Step 3] Sending Turn 1: Literature Search Request...")
    t1_payload = {
        "conversation_id": cid,
        "message": "Find recent studies on eye movement biomarkers in Alzheimer disease"
    }
    t1_resp = client.post("/chat", json=t1_payload)
    assert t1_resp.status_code == 200, f"Turn 1 failed: {t1_resp.text}"
    t1_data = t1_resp.json()
    print(f" -> Tools Invoked: {t1_data['tools_used']}")
    print(f" -> Sources Count: {len(t1_data['sources'])}")
    print(f" -> Response Preview:\n    {t1_data['response'][:160]}...")

    # 4. Send Second Message (Requires Multi-Turn Coreference Resolution)
    print("\n[Step 4] Sending Turn 2: Follow-up with Coreference ('it')...")
    t2_payload = {
        "conversation_id": cid,
        "message": "How is it different from Parkinson disease?"
    }
    t2_resp = client.post("/chat", json=t2_payload)
    assert t2_resp.status_code == 200, f"Turn 2 failed: {t2_resp.text}"
    t2_data = t2_resp.json()
    print(f" -> Tools Invoked: {t2_data['tools_used']}")
    print(f" -> Response Preview:\n    {t2_data['response'][:160]}...")

    # 5. Verify Relational Persistence in PostgreSQL
    print("\n[Step 5] Directly Querying PostgreSQL to Verify Relational Persistence...")
    db = SessionLocal()
    try:
        db_conv = db.query(Conversation).filter(Conversation.id == cid).first()
        assert db_conv is not None, "Conversation record not found in PostgreSQL!"
        db_msgs = db.query(Message).filter(Message.conversation_id == cid).order_by(Message.id.asc()).all()
        assert len(db_msgs) == 4, f"Expected 4 persisted messages in DB, got {len(db_msgs)}"

        print(f" -> PostgreSQL Conversation: ID={db_conv.id}, Title='{db_conv.title}'")
        print(f" -> Stored Messages Count: {len(db_msgs)}")
        for idx, m in enumerate(db_msgs, 1):
            tool_note = f" (tool_metadata={m.tool_calls})" if m.tool_calls else ""
            print(f"    Turn {idx} [{m.role}]: {m.content[:55]}...{tool_note}")
    finally:
        db.close()

    print("\n" + "=" * 65)
    print(" ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
