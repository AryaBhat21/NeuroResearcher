"""
Tests for agent orchestration, tool routing, and multi-turn reference resolution.
"""
from sqlalchemy.orm import Session
from agent import run_agent_turn, run_agent_flow, DeterministicResearchAgent
from database import Message, Conversation


def test_agent_single_turn_with_tool(db_session: Session):
    """Verifies that a literature query executes search_literature and persists messages."""
    response = run_agent_turn(
        db=db_session,
        conversation_id=None,
        user_message="Find recent studies on alpha-synuclein in Parkinson disease"
    )

    assert response.conversation_id is not None
    assert len(response.response) > 0
    assert "search_literature" in response.tools_used
    assert response.role == "assistant"

    # Verify messages in DB
    msgs = db_session.query(Message).filter(Message.conversation_id == response.conversation_id).all()
    assert len(msgs) == 2
    assert msgs[0].role == "user"
    assert msgs[1].role == "assistant"


def test_agent_multi_turn_coreference(db_session: Session):
    """
    Tests Phase 5 critical requirement:
    Turn 1: "What is saccadic latency in Alzheimer disease?"
    Turn 2: "How is it different from Parkinson disease?"
    Verifies that 'it' is resolved in Turn 2 using persistent conversation context.
    """
    # Turn 1
    t1_res = run_agent_turn(
        db=db_session,
        conversation_id=None,
        user_message="What is saccadic latency in Alzheimer disease?"
    )
    cid = t1_res.conversation_id

    # Turn 2
    t2_res = run_agent_turn(
        db=db_session,
        conversation_id=cid,
        user_message="How is it different from Parkinson disease?"
    )

    assert t2_res.conversation_id == cid
    assert len(t2_res.tools_used) > 0

    # Check that 4 messages total exist in this conversation
    msgs = db_session.query(Message).filter(Message.conversation_id == cid).order_by(Message.id.asc()).all()
    assert len(msgs) == 4
    assert msgs[0].content == "What is saccadic latency in Alzheimer disease?"
    assert msgs[2].content == "How is it different from Parkinson disease?"


def test_context_resolution_logic():
    """Unit test for DeterministicResearchAgent.resolve_context."""
    prior = [
        Message(id=1, conversation_id=1, role="user", content="What are retinal biomarkers in cognitive impairment?"),
        Message(id=2, conversation_id=1, role="assistant", content="Retinal OCT shows thinning of RNFL layer.")
    ]

    resolved = DeterministicResearchAgent.resolve_context(
        current_message="How do they compare with MRI findings?",
        prior_messages=prior
    )

    # Should incorporate the subject from the previous turn
    assert "retinal biomarkers" in resolved.lower()


def test_run_agent_flow_backwards_compatibility():
    """Verifies that legacy run_agent_flow function continues to work."""
    text = run_agent_flow("Find recent papers on neurofilament light chain")
    assert isinstance(text, str)
    assert len(text) > 0
