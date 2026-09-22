"""
Pydantic schemas for request validation and response serialization.
Complies with Pydantic v2 standards with ConfigDict(from_attributes=True).
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Research Query Schemas (Phase 1 & 2)
# ---------------------------------------------------------------------------

class ResearchQueryCreate(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000, description="Biomedical research query string")


class ResearchQueryRead(BaseModel):
    id: int
    query: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Message & Conversation Schemas (Phase 3 & 4)
# ---------------------------------------------------------------------------

class MessageRead(BaseModel):
    id: int
    conversation_id: int
    role: str
    content: str
    tool_calls: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ConversationCreate(BaseModel):
    title: Optional[str] = Field(None, max_length=255, description="Optional topic or title for the conversation")


class ConversationRead(BaseModel):
    id: int
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    message_count: Optional[int] = 0

    model_config = ConfigDict(from_attributes=True)


class ConversationDetail(BaseModel):
    id: int
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    messages: List[MessageRead] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Chat Request & Response Schemas (Agent Execution)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User question or research prompt")
    conversation_id: Optional[int] = Field(None, description="ID of existing conversation to continue, or None to create a new session")


class ChatResponse(BaseModel):
    conversation_id: int
    response: str
    role: str = "assistant"
    tools_used: List[str] = Field(default_factory=list, description="Names of tools executed during this turn")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Structured citation records retrieved from tools")


# ---------------------------------------------------------------------------
# System & Health Schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    database: str
    agent_mode: str
    timestamp: datetime
