"""
Database configuration and SQLAlchemy ORM models for NeuroResearch.
Manages connections, session lifecycles, and persistent storage for
research queries, conversations, and message history.
"""
import os
import logging
from typing import Generator, Tuple
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from sqlalchemy.sql import func

load_dotenv()

logger = logging.getLogger("neuro_research.database")

# Database URL loaded strictly from environment with a local fallback
DEFAULT_DB_URL = "postgresql://postgres:postgres@localhost:5432/neuro_research"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

# SQLAlchemy engine with connection pool pre-ping to handle stale connections gracefully
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class ResearchQuery(Base):
    """
    Stores standalone literature research queries.
    Preserved for backwards compatibility with existing Phase 1 & 2 records.
    """
    __tablename__ = "research_queries"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Conversation(Base):
    """
    Represents an ongoing multi-turn research dialogue session.
    """
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Cascading relationship: deleting a conversation deletes all associated messages
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.id"
    )


class Message(Base):
    """
    Individual turn within a multi-turn conversation.
    Supports user prompts, assistant answers, and tool execution metadata.
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String, nullable=False)  # 'user', 'assistant', 'tool', 'system'
    content = Column(Text, nullable=False)
    tool_calls = Column(Text, nullable=True)  # JSON-encoded string of tool calls/results
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    conversation = relationship("Conversation", back_populates="messages")


# ---------------------------------------------------------------------------
# Database Utilities & Lifespan Management
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all database tables if they do not already exist."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")
    except Exception as exc:
        logger.error(f"Failed to initialize database tables: {exc}")
        raise


def check_db_connection() -> Tuple[bool, str]:
    """
    Verifies that the database engine can successfully connect and query.
    Returns (True, 'Connected') or (False, error_detail).
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Connected"
    except Exception as exc:
        return False, str(exc)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency yielding a thread-safe database session per request.
    Automatically rolls back uncommitted transactions on unhandled exceptions
    and always closes the session in the finally block.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
