"""
Tests for database models, session management, and persistence lifecycles.
"""
from sqlalchemy.orm import Session
from database import ResearchQuery, Conversation, Message


def test_research_query_persistence(db_session: Session):
    """Verifies that ResearchQuery records can be added and queried."""
    query_obj = ResearchQuery(query="Saccadic eye movement biomarkers")
    db_session.add(query_obj)
    db_session.commit()

    retrieved = db_session.query(ResearchQuery).filter(ResearchQuery.query == "Saccadic eye movement biomarkers").first()
    assert retrieved is not None
    assert retrieved.id is not None
    assert retrieved.query == "Saccadic eye movement biomarkers"


def test_conversation_and_message_cascade(db_session: Session):
    """Verifies conversation creation, message association, and cascade delete."""
    conv = Conversation(title="Dementia Study")
    db_session.add(conv)
    db_session.commit()
    db_session.refresh(conv)

    msg1 = Message(conversation_id=conv.id, role="user", content="Tell me about tau PET")
    msg2 = Message(conversation_id=conv.id, role="assistant", content="Tau PET imaging measures neurofibrillary tangles.")
    db_session.add_all([msg1, msg2])
    db_session.commit()

    # Query messages
    messages = db_session.query(Message).filter(Message.conversation_id == conv.id).order_by(Message.id.asc()).all()
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"

    # Delete conversation and ensure messages are deleted via cascade
    conv_id = conv.id
    db_session.delete(conv)
    db_session.commit()

    orphaned = db_session.query(Message).filter(Message.conversation_id == conv_id).all()
    assert len(orphaned) == 0


def test_message_chronological_ordering(db_session: Session):
    """Verifies that messages are strictly ordered chronologically by ID."""
    conv = Conversation(title="Timeline Test")
    db_session.add(conv)
    db_session.commit()

    for i in range(5):
        msg = Message(conversation_id=conv.id, role="user" if i % 2 == 0 else "assistant", content=f"Message {i}")
        db_session.add(msg)
    db_session.commit()

    ordered = db_session.query(Message).filter(Message.conversation_id == conv.id).order_by(Message.id.asc()).all()
    contents = [m.content for m in ordered]
    assert contents == [f"Message {i}" for i in range(5)]
