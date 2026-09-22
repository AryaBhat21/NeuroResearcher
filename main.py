"""
NeuroLens FastAPI Application Backend
Provides research query storage, health checks, and AI agent endpoints.
"""
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from dotenv import load_dotenv
from agent import run_agent_flow

# Load environment variables from .env file
load_dotenv()


# Database configuration
# Replace "postgres:postgres" with your actual Postgres username:password if different
DATABASE_URL = "postgresql://postgres:arya21bhat@localhost:5432/neuro_research"

# The engine handles connection pooling and communication with the Postgres driver
engine = create_engine(DATABASE_URL)

# SessionLocal is our session factory; each instance is a single database transaction session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for our ORM models (corresponds to tables in Postgres)
Base = declarative_base()

# Dependency to get a database session per HTTP request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# SQLAlchemy database model representing the 'research_queries' table
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func

class ResearchQuery(Base):
    __tablename__ = "research_queries"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# Automatically create tables in PostgreSQL on application startup
Base.metadata.create_all(bind=engine)

# Pydantic model for incoming request validation
class ResearchQueryCreate(BaseModel):
    query: str


# Initialize the main FastAPI application instance
app = FastAPI(title="Neurology Research Workflow Backend")

@app.get("/health")
def health_check():
    """
    Simple health check endpoint to verify the server is running.
    """
    return {"status": "healthy"}


@app.post("/research-queries")
def create_research_query(payload: ResearchQueryCreate, db: Session = Depends(get_db)):
    """
    Endpoint to receive a new research query and save it to the database.
    """
    # 1. Instantiate the ORM object from Pydantic data
    new_query = ResearchQuery(query=payload.query)
    
    # 2. Add the object to the session transaction
    db.add(new_query)
    
    # 3. Commit the transaction to save to Postgres
    db.commit()
    
    # 4. Refresh the instance to load generated DB values (like id and created_at)
    db.refresh(new_query)
    
    # 5. Return the ORM object (FastAPI will automatically serialize it to JSON)
    return new_query


@app.get("/research-queries")
def get_all_research_queries(db: Session = Depends(get_db)):
    """
    Endpoint to retrieve all research queries from the database.
    """
    # Query the database to retrieve all rows from the 'research_queries' table
    queries = db.query(ResearchQuery).all()
    
    # Return the list of ORM objects (FastAPI will automatically serialize this to a JSON list)
    return queries

@app.get("/research-queries/{id}")
def get_research_query_by_id(id: int, db: Session = Depends(get_db)):
    query = db.query(ResearchQuery).filter(ResearchQuery.id == id).first()
    if query is None:
        raise HTTPException(status_code=404, detail="Research query not found")
    return query        

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat_with_agent(payload: ChatRequest):
    """
    Endpoint to interact with the agent flow (LLM + local tool calling).
    """
    try:
        response_text = run_agent_flow(payload.message)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



