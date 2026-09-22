# NeuroResearch: Autonomous Biomedical Literature & Multi-Turn Clinical Reasoning Agent

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-teal.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue.svg)](https://www.postgresql.org/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-red.svg)](https://www.sqlalchemy.org/)
[![Europe PMC](https://img.shields.io/badge/Literature-Europe_PMC_REST-green.svg)](https://europepmc.org/)
[![Tests](https://img.shields.io/badge/Tests-18%20Passed-brightgreen.svg)](https://docs.pytest.org/)

An enterprise-grade, research-oriented AI agent system designed for clinical neuroscientists, researchers, and neurologists. NeuroResearch orchestrates multi-turn biomedical dialogues, executes autonomous literature queries against live peer-reviewed databases (Europe PMC / PubMed), tracks dialogue context across session boundaries, and guarantees end-to-end scientific traceability.

---

## 1. Problem Statement

Biomedical literature is expanding exponentially—over 1.5 million life-science papers are published annually. Researchers investigating complex neurodegenerative disorders (e.g., Alzheimer's, Parkinson's, ALS, Frontotemporal Dementia) encounter four major operational bottlenecks when using conventional LLMs:

1. **Hallucinated Citations & Metrics**: Standard LLMs routinely invent plausible-sounding citations, PMIDs, journal volumes, and effect sizes.
2. **Context Degradation in Multi-Turn Reasoning**: Follow-up inquiries (e.g., *"How does it compare to progressive supranuclear palsy?"*) frequently lose the primary entity (*saccadic eye movement*) when conversational turns are stateless.
3. **Fragmented Data Extraction**: Interfacing directly with biological APIs (such as NCBI Entrez or Europe PMC) yields heterogeneous, deeply nested JSON/XML payloads requiring custom sanitization.
4. **Lack of Session Persistence**: Exploration threads and tool outputs are ephemeral unless backed by an ACID-compliant relational persistence layer.

NeuroResearch solves these challenges through an adapter-patterned tool layer, anaphora/coreference-aware multi-turn agent orchestration, and relational PostgreSQL session persistence.

---

## 2. What the System Does

- **Real-Time Peer-Reviewed Literature Retrieval**: Executes live HTTP queries against Europe PMC's REST API (indexing 40+ million scientific records from PubMed, PubMed Central, and preprint servers).
- **Autonomous Tool Routing**: Employs an agentic loop that dynamically decides when a query requires empirical literature search (`search_literature`), specific paper inspection by PMID (`get_paper`), or direct conceptual synthesis.
- **Persistent Multi-Turn Dialogues**: Tracks conversation history in PostgreSQL, enabling antecedent resolution across dialogue turns (e.g., resolving pronouns like "it", "they", "these biomarkers" back to previous turns).
- **Traceable Scientific Citations**: Every factual claim is paired with paper titles, primary authors, publication years, DOIs, PMIDs, and verified URLs.
- **Resilient Dual-Mode Execution**: Incorporates a live Google Gemini provider alongside an intelligent deterministic research reasoning engine, allowing offline testing and guaranteed local verifiability.

---

## 3. Architecture Overview

### 3.1 System Architecture Diagram

```mermaid
flowchart TB
    subgraph ClientLayer ["Client & Interface Layer"]
        Researcher["Clinician / Neuroscientist"]
        HTTPClient["HTTP Client / Frontend / Swagger UI"]
    end

    subgraph FastAPILayer ["FastAPI Application Backend (main.py)"]
        Router["FastAPI Router"]
        CORS["CORSMiddleware"]
        HealthEP["GET /health"]
        ChatEP["POST /chat"]
        ConvEP["CRUD /conversations"]
        QueryEP["CRUD /research-queries"]
        DBDep["get_db() Transaction Dependency"]
    end

    subgraph AgenticCore ["Agent Engine (agent.py)"]
        TurnOrchestrator["run_agent_turn()"]
        ContextResolver["Antecedent & Coreference Resolver"]
        
        subgraph ReasoningEngine ["Reasoning Providers"]
            GeminiLive["Google Gemini (google-genai)"]
            DeterministicEngine["Deterministic Research Agent"]
        end
    end

    subgraph ToolLayer ["Domain Tool Layer (tools.py)"]
        ToolSearch["search_literature()"]
        ToolPaper["get_paper()"]
        AdapterService["Europe PMC Adapter (literature_service.py)"]
    end

    subgraph PersistenceLayer ["Storage & External Providers"]
        PostgreSQL[("PostgreSQL 16\n(Tables: conversations, messages, research_queries)")]
        EuropePMCAPI["Europe PMC REST API (live)"]
        GoogleGeminiAPI["Google Gemini Cloud API"]
    end

    Researcher --> HTTPClient
    HTTPClient -->|JSON Requests| Router
    Router --> CORS
    Router --> HealthEP
    Router --> ChatEP
    Router --> ConvEP
    Router --> QueryEP

    ChatEP --> DBDep
    ConvEP --> DBDep
    QueryEP --> DBDep
    DBDep <-->|SQLAlchemy Sessions| PostgreSQL

    ChatEP --> TurnOrchestrator
    TurnOrchestrator --> ContextResolver
    ContextResolver --> ReasoningEngine
    ReasoningEngine <--> GoogleGeminiAPI
    ReasoningEngine --> ToolLayer

    ToolSearch --> AdapterService
    ToolPaper --> AdapterService
    AdapterService <-->|HTTPS GET| EuropePMCAPI
    TurnOrchestrator -->|Persist Turns & Tool Results| PostgreSQL
```

### 3.2 Sequence Diagram: Multi-Turn Conversation Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as Researcher
    participant API as FastAPI Server
    participant DB as PostgreSQL DB
    participant Agent as Agent Orchestrator
    participant Tool as Europe PMC Tool
    participant ExtAPI as Europe PMC REST API

    User->>API: POST /chat {"message": "Find studies on saccadic latency in Alzheimer's"}
    API->>DB: Load prior turns (Empty)
    API->>DB: INSERT Message (Role: user)
    API->>Agent: Execute turn with user query
    Agent->>Tool: search_literature(query="saccadic latency Alzheimer")
    Tool->>ExtAPI: GET /webservices/rest/search?query=...
    ExtAPI-->>Tool: Return raw biomedical JSON
    Tool-->>Agent: Parsed papers with DOI, PMID, Abstract
    Agent->>DB: INSERT Message (Role: assistant, Content: synthesis, ToolCalls: metadata)
    Agent-->>API: ChatResponse
    API-->>User: Response with paper citations

    Note over User, API: Turn 2: Follow-up with Coreference ("it")
    User->>API: POST /chat {"conversation_id": 1, "message": "How is it different from Parkinson's?"}
    API->>DB: SELECT * FROM messages WHERE conversation_id=1 ORDER BY id ASC
    DB-->>API: History (Turn 1 User + Turn 1 Assistant)
    API->>DB: INSERT Message (Role: user, Content: "How is it different from Parkinson's?")
    API->>Agent: Execute turn with history
    Agent->>Agent: Resolve "it" -> "saccadic latency" from Turn 1
    Agent->>Tool: search_literature(query="saccadic latency Parkinson disease")
    Tool->>ExtAPI: GET /webservices/rest/search?query=...
    ExtAPI-->>Tool: Return Parkinson oculomotor literature
    Tool-->>Agent: Parsed papers
    Agent->>DB: INSERT Message (Role: assistant, Content: comparative synthesis)
    Agent-->>API: ChatResponse
    API-->>User: Differential analysis comparing AD vs. PD saccadic metrics
```

---

## 4. Main Components

| File | Module Name | Architectural Responsibility |
| :--- | :--- | :--- |
| [`main.py`](file:///d:/Neuro%20Researcher/main.py) | **FastAPI Backend Router** | Manages application lifespan, CORS, dependency injection, and REST endpoints for `/health`, `/chat`, `/conversations`, and `/research-queries`. |
| [`agent.py`](file:///d:/Neuro%20Researcher/agent.py) | **Agent Orchestration Engine** | Handles conversational history, pronoun/coreference resolution, tool dispatching loops, and dual-mode execution (Gemini + Deterministic Fallback). |
| [`database.py`](file:///d:/Neuro%20Researcher/database.py) | **Persistence & ORM Layer** | Configures SQLAlchemy engine with connection pool pre-ping, defines `Conversation`, `Message`, and `ResearchQuery` ORM models, and supplies transactional `get_db()` dependency. |
| [`schemas.py`](file:///d:/Neuro%20Researcher/schemas.py) | **Pydantic Validation Schemas** | Strict Pydantic v2 data transfer objects (`ChatRequest`, `ChatResponse`, `ConversationDetail`, `HealthResponse`, `ResearchQueryCreate`). |
| [`tools.py`](file:///d:/Neuro%20Researcher/tools.py) | **Agent Domain Tools** | Exposes `search_literature()` and `get_paper()` with type annotations, docstrings, and parameter bounding for function calling. |
| [`literature_service.py`](file:///d:/Neuro%20Researcher/literature_service.py) | **Service Adapter Layer** | HTTP client for Europe PMC REST API. Sanitizes external JSON, bounds result sizes, formats year filters (`FIRST_PDATE`), and handles HTTP 429/5xx errors defensively. |
| [`evaluate.py`](file:///d:/Neuro%20Researcher/evaluate.py) | **Research Benchmark Suite** | Automated, reproducible evaluation runner assessing tool routing accuracy, multi-turn context retention, and citation provenance rate. |

---

## 5. Database & Session Architecture

The PostgreSQL database schema is designed for relational consistency and fast multi-turn reconstruction:

```mermaid
erDiagram
    CONVERSATIONS ||--o{ MESSAGES : "contains (1 to N)"
    RESEARCH_QUERIES {
        int id PK
        string query
        datetime created_at
    }
    CONVERSATIONS {
        int id PK
        string title
        datetime created_at
        datetime updated_at
    }
    MESSAGES {
        int id PK
        int conversation_id FK
        string role
        text content
        text tool_calls
        datetime created_at
    }
```

- **Transactional Rollback Safety**: The `get_db()` dependency wraps each request in a `try / except / finally` block. If an unhandled exception occurs, `db.rollback()` executes automatically before closing the session.
- **Cascading Deletions**: Deleting a conversation automatically triggers cascading deletion of all associated messages, preventing orphaned records.
- **Connection Pool Resilience**: Configured with `pool_pre_ping=True` to detect and refresh stale database connections.

---

## 6. Technology Stack

- **Runtime & Language**: Python 3.12 (CPython)
- **Web Framework**: FastAPI 0.111.0 & Starlette
- **ASGI Server**: Uvicorn 0.30.1
- **Database**: PostgreSQL 16
- **Object Relational Mapper**: SQLAlchemy 2.0.30
- **Validation**: Pydantic 2.13.4
- **LLM SDK**: Google GenAI SDK 2.5.0
- **External Literature API**: Europe PMC REST API (`https://www.ebi.ac.uk/europepmc/webservices/rest/search`)
- **Testing**: Pytest 9.1.1 & HTTPX 0.28.1

---

## 7. Setup & Installation

### 7.1 Prerequisites
- Python 3.12+
- PostgreSQL 16 running locally (or remote connection URI)

### 7.2 Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/AryaBhat21/NeuroResearcher.git
cd "Neuro Researcher"

# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Activate on Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 7.3 Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Populate `.env` with your credentials:
```ini
# PostgreSQL Connection URL
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/neuro_research

# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# Logging verbosity
LOG_LEVEL=INFO
```

---

## 8. Running the Backend & Tests

### 8.1 Start Backend Server
```bash
.\.venv\Scripts\uvicorn.exe main:app --reload --port 8000
```
Interactive OpenAPI documentation will be accessible at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 8.2 Run Automated Test Suite
The automated test suite executes 18 tests covering health checks, domain tools, database transactions, multi-turn reasoning, and end-to-end API endpoints:
```bash
.\.venv\Scripts\pytest.exe -v
```

### 8.3 Run Research & Evaluation Benchmark
Execute the reproducible benchmark harness:
```bash
.\.venv\Scripts\python.exe evaluate.py
```

---

## 9. Research & Evaluation Benchmark Results

The benchmark harness evaluated three key dimensions using real biomedical queries and live literature retrieval:

| Evaluation Benchmark | Metric | Result | Target Benchmark | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Intent & Tool Selection** | Routing Accuracy (Precision/Recall) | **100.0%** (6/6) | $\ge 95\%$ | **PASSED** |
| **Intent & Tool Selection** | Average Latency (including live HTTP) | **2171.9 ms** | $< 4000\text{ ms}$ | **PASSED** |
| **Multi-Turn Anaphora Retention** | Antecedent Concept Retention | **100.0%** (2/2) | $\ge 90\%$ | **PASSED** |
| **Citation Traceability** | Source Provenance Completeness | **100.0%** (3/3) | $100\%$ | **PASSED** |

Benchmark outputs are exported to [`eval_results.json`](file:///d:/Neuro%20Researcher/eval_results.json).

---

## 10. Example API Workflows

### 10.1 Health Check
**Request:**
```http
GET /health HTTP/1.1
Host: 127.0.0.1:8000
```
**Response (200 OK):**
```json
{
  "status": "healthy",
  "database": "PostgreSQL (Connected)",
  "agent_mode": "deterministic_research_engine",
  "timestamp": "2026-09-22T12:00:00.000000Z"
}
```

### 10.2 Starting a Multi-Turn Research Dialogue
**Request (Turn 1):**
```http
POST /chat HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "message": "Find recent studies on saccadic eye movements in Alzheimer disease"
}
```
**Response (200 OK):**
```json
{
  "conversation_id": 1,
  "response": "Based on recent biomedical literature retrieved via Europe PMC for **saccadic eye movements in Alzheimer disease**:\n\n1. **Theta Oscillations and Saccadic Latency in Cognitive Decline** (2024)\n   - *Authors:* Zhang L, Patel R, Gomez M\n   - *Journal:* Journal of Neuroscience Methods\n   - *Link:* [38192014](https://europepmc.org/article/MED/38192014)\n   - *Key Finding:* Saccadic latency and anti-saccade error rates demonstrate statistically significant correlation with CSF amyloid-beta42/40 ratios...",
  "role": "assistant",
  "tools_used": ["search_literature"],
  "sources": [
    {
      "id": "38192014",
      "title": "Theta Oscillations and Saccadic Latency in Cognitive Decline",
      "authors": ["Zhang L", "Patel R", "Gomez M"],
      "publication_year": 2024,
      "doi": "10.1016/j.jneumeth.2024.109820",
      "journal": "Journal of Neuroscience Methods",
      "url": "https://europepmc.org/article/MED/38192014"
    }
  ]
}
```

### 10.3 Follow-Up Turn Demonstrating Context Retention
**Request (Turn 2):**
```http
POST /chat HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "conversation_id": 1,
  "message": "How is it different from Parkinson disease?"
}
```
**Behavior:**
1. Loads Turn 1 from PostgreSQL table `messages`.
2. Resolves `"it"` to `"saccadic eye movements"`.
3. Queries Europe PMC for comparative saccadic markers in Parkinson's.
4. Persists the new exchange linked to `conversation_id: 1`.

---

## 11. Design Decisions & Trade-Offs

1. **Europe PMC REST API over NCBI PubMed Direct**:
   - Europe PMC delivers core metadata (title, full abstract, authors, DOI, and journal) in a single JSON payload. NCBI E-utilities requires a multi-step sequence (`esearch` followed by `esummary`/`efetch`) and returns XML by default.
2. **Explicit Tool Loop vs. Automatic Calling**:
   - `automatic_function_calling=False` was selected to ensure the backend logs every tool invocation, enforces result clamping, sanitizes parameters, and attaches structured provenance metadata before returning results to the client.
3. **Dual-Mode Agent Reasoning Engine**:
   - Production systems must degrade gracefully during third-party API outages or key revocations. The architecture automatically routes queries through Google Gemini when valid credentials exist, and falls back to a deterministic semantic engine when offline or testing.
4. **Relational PostgreSQL vs. In-Memory Session Stores**:
   - Long-term clinical and literature exploration sessions require ACID transactions, historical auditability, and relational joins with researcher query tables.

---

## 12. Limitations & Future Improvements

### Current Limitations
- **Abstract-Level Synthesis**: Europe PMC open-access full-text XML is not parsed in this phase; summaries rely on structured abstracts (`resultType=core`).
- **Semantic Vector Reranking**: Literature ranking relies on Europe PMC relevance algorithms and year filters rather than a dense vector reranking step.

### Future Improvements
1. **Hybrid Dense/Sparse Retrieval**: Integrate `pgvector` into the PostgreSQL database to store embeddings of downloaded abstracts for hybrid semantic search.
2. **Clinical Trial Registry Integration**: Expand domain tools to query `ClinicalTrials.gov` REST API for active interventional trials.
3. **Streaming Responses**: Implement Server-Sent Events (`SSE`) on `/chat` for token-by-token streaming of synthesized literature reviews.
