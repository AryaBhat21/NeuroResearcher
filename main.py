"""
NeuroResearch FastAPI Application Backend.
Provides biomedical research query endpoints, multi-turn conversational AI sessions,
tool execution orchestration, and health monitoring.
"""
import os
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from database import init_db, get_db, check_db_connection, ResearchQuery, Conversation, Message
from schemas import (
    ResearchQueryCreate,
    ResearchQueryRead,
    ConversationCreate,
    ConversationRead,
    ConversationDetail,
    MessageRead,
    ChatRequest,
    ChatResponse,
    HealthResponse,
)
from agent import run_agent_turn, _gemini_available, get_gemini_client

# Load environment configuration
load_dotenv()

# Configure root logger
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("neuro_research.api")


# ---------------------------------------------------------------------------
# Lifespan Management
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Initializes database tables on startup and verifies system dependencies.
    """
    logger.info("Initializing NeuroResearch backend services...")
    try:
        init_db()
        db_ok, db_msg = check_db_connection()
        if db_ok:
            logger.info("Database connectivity established successfully.")
        else:
            logger.warning(f"Database connectivity warning: {db_msg}")
    except Exception as exc:
        logger.error(f"Startup database initialization error: {exc}")

    # Check initial LLM client status
    gemini_client = get_gemini_client()
    if gemini_client:
        logger.info("Gemini LLM client initialized.")
    else:
        logger.info("Deterministic research agent initialized as active reasoning engine.")

    yield

    logger.info("Shutting down NeuroResearch backend services...")


# ---------------------------------------------------------------------------
# FastAPI Application Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NeuroResearch AI Agent API",
    description=(
        "Production-grade research backend for biomedical literature retrieval, "
        "neurodegenerative biomarker discovery, and multi-turn clinical dialogue."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Enable Cross-Origin Resource Sharing (CORS) for frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# System & Health Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Verifies API availability, database connectivity, and active agent execution mode.
    """
    db_connected, db_status = check_db_connection()
    agent_mode = "gemini_live" if _gemini_available is True else "deterministic_research_engine"

    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        database=f"PostgreSQL ({db_status})",
        agent_mode=agent_mode,
        timestamp=datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Conversational Agent Endpoints (Phase 4 & Multi-Turn)
# ---------------------------------------------------------------------------

@app.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["Agent Dialogue"]
)
def chat_with_agent(payload: ChatRequest, db: Session = Depends(get_db)):
    """
    Main conversational endpoint:
    - Accepts a research query and optional conversation_id.
    - Loads prior context from PostgreSQL to resolve multi-turn references.
    - Selects and runs domain tools (e.g. Europe PMC literature search, paper retrieval).
    - Persists both user and assistant dialogue turns with tool execution metadata.
    """
    try:
        chat_response = run_agent_turn(
            db=db,
            conversation_id=payload.conversation_id,
            user_message=payload.message
        )
        return chat_response
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Unhandled error in /chat: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your research request."
        )


@app.post(
    "/conversations",
    response_model=ConversationRead,
    status_code=status.HTTP_201_CREATED,
    tags=["Conversations"]
)
def create_conversation(payload: ConversationCreate, db: Session = Depends(get_db)):
    """
    Creates a new research conversation session.
    """
    conv = Conversation(title=payload.title or "New Research Session")
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return ConversationRead(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        message_count=0
    )


@app.get(
    "/conversations",
    response_model=List[ConversationRead],
    tags=["Conversations"]
)
def list_conversations(db: Session = Depends(get_db)):
    """
    Lists all research conversations ordered by most recently updated.
    """
    conversations = db.query(Conversation).order_by(Conversation.updated_at.desc()).all()
    results = []
    for c in conversations:
        count = db.query(Message).filter(Message.conversation_id == c.id).count()
        results.append(
            ConversationRead(
                id=c.id,
                title=c.title,
                created_at=c.created_at,
                updated_at=c.updated_at,
                message_count=count
            )
        )
    return results


@app.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetail,
    tags=["Conversations"]
)
def get_conversation_history(conversation_id: int, db: Session = Depends(get_db)):
    """
    Retrieves full conversation details and its complete chronological message history.
    """
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} was not found."
        )

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.id.asc())
        .all()
    )

    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=[MessageRead.model_validate(m) for m in messages]
    )


@app.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Conversations"]
)
def delete_conversation(conversation_id: int, db: Session = Depends(get_db)):
    """
    Deletes a conversation session and cascades deletion to all associated messages.
    """
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} was not found."
        )
    db.delete(conv)
    db.commit()
    return None


# ---------------------------------------------------------------------------
# Research Query Endpoints (Phase 1 & 2 Backwards Compatibility)
# ---------------------------------------------------------------------------

@app.post(
    "/research-queries",
    response_model=ResearchQueryRead,
    status_code=status.HTTP_201_CREATED,
    tags=["Research Queries"]
)
def create_research_query(payload: ResearchQueryCreate, db: Session = Depends(get_db)):
    """
    Saves a research query string to the database.
    """
    new_query = ResearchQuery(query=payload.query)
    db.add(new_query)
    db.commit()
    db.refresh(new_query)
    return new_query


@app.get(
    "/research-queries",
    response_model=List[ResearchQueryRead],
    tags=["Research Queries"]
)
def get_all_research_queries(db: Session = Depends(get_db)):
    """
    Retrieves all research queries stored in the database.
    """
    return db.query(ResearchQuery).order_by(ResearchQuery.id.desc()).all()


@app.get(
    "/research-queries/{id}",
    response_model=ResearchQueryRead,
    tags=["Research Queries"]
)
def get_research_query_by_id(id: int, db: Session = Depends(get_db)):
    """
    Retrieves a single research query by primary key ID.
    """
    query = db.query(ResearchQuery).filter(ResearchQuery.id == id).first()
    if query is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research query with ID {id} not found"
        )
    return query
