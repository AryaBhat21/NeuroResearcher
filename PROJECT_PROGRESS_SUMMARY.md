# Neurology Research & Workflow Agent - Complete Learning & Architecture Summary

---

## Section 1: Key Concepts & Learnings Mastered (Start Here)

This section documents all foundational concepts, mental models, environment rules, and exercises covered throughout our sessions.

---

### 1.1 The "Restaurant Analogy" of Backend & Agent Architecture
To understand how an AI-powered backend operates, we use the restaurant analogy:

```
[ Diner (User) ] ───(Sends Request)───► [ Host / Waiter (FastAPI Server) ]
                                                   │
                                                   ▼
                                     [ Head Waiter (Gemini LLM) ]
                                                   │
                                      (Requests Tool Execution)
                                                   ▼
                                      [ Kitchen Chefs (Python Tools) ]
                                                   │
                                        (Fetches Ingredients)
                                                   ▼
                                      [ Pantry / Storage (PostgreSQL) ]
```

* **Diner (User / Client)**: Submits a query (e.g., *"Find recent studies on eye movement in Alzheimer's"*).
* **Host / Front Desk (FastAPI)**: Accepts HTTP requests (`POST /chat`, `POST /research-queries`), validates payload format, and manages session state.
* **Head Waiter (Gemini LLM)**: Decides whether the request can be answered immediately or requires fetching external data from the kitchen. If external data is needed, Gemini emits a structured JSON tool request.
* **Kitchen Chefs (Local Python Functions in `tools.py`)**: Executes the logic (`search_literature`, `get_paper`) locally on the server.
* **Pantry / Storage (PostgreSQL Database)**: Persists raw data securely using SQLAlchemy ORM.

---

### 1.2 LLM Tool Calling Mechanics & Key Takeaways

#### **Why LLMs Don't Execute Python Code Directly**
* Large Language Models are text/token predictors; they do **not** run code on your computer.
* When Gemini determines a tool call is needed, it stops generating prose and returns a **Structured JSON Tool Call Object** containing:
  1. The target function name (`"search_literature"`).
  2. The extracted arguments (`{"query": "eye-movement Alzheimer"}`).
* Your backend Python program reads this JSON, executes the actual Python function locally, receives the result, converts it into a `function_response` message, and sends it back to Gemini.

#### **Manual Tool Execution Loop vs. Automatic Execution**
In our codebase ([`agent.py`](file:///d:/Neuro%20Researcher/agent.py) & [`test_llm.py`](file:///d:/Neuro%20Researcher/test_llm.py)), we explicitly set `automatic_function_calling=AutomaticFunctionCallingConfig(disable=True)`.
* **Reason**: Disabling auto-calling allows our backend full control to log tool execution (`[TOOL EXECUTION]`), validate arguments, restrict permissions, and inspect raw intermediate payloads before returning results back to Gemini.

#### **Summary of Exercises & Quiz Answers**
* **JSON Serialization**: Tool outputs must be formatted as dictionaries/JSON objects so the LLM can interpret key-value structures.
* **Tool Decision Criteria**: The LLM compares the user's intent against docstrings and argument type hints of registered tools. If no tool matches, it responds directly using pre-trained knowledge.
* **Message History Integrity**: The conversation history sent back to Gemini must include:
  1. Original user query (`role: user`).
  2. Model candidate response containing the tool call (`role: model`).
  3. Function execution result (`role: user` / `part: function_response`).

---

### 1.3 Backend & Database Core Concepts

#### **Frontend vs. Backend vs. Database**
| Component | Primary Responsibility | Technology Used |
| :--- | :--- | :--- |
| **Frontend** | User Interface (Buttons, forms, rendering text) | React / HTML / Streamlit |
| **Backend** | Business logic, LLM orchestration, security, routing | Python 3.12, FastAPI, Uvicorn |
| **Database** | Long-term structured storage & persistence | PostgreSQL 16 |

#### **Pydantic Models vs. SQLAlchemy ORM Models**
* **Pydantic (`BaseModel`)**: Validates incoming HTTP requests coming over the network (e.g. `ResearchQueryCreate`). Performs automatic type conversion (e.g., string `"101"` $\rightarrow$ int `101`).
* **SQLAlchemy (`Base`)**: Maps Python objects directly to database tables (`research_queries`). Manages SQL connection pools, transactions (`db.commit()`), and auto-generated fields (`id`, `created_at`).

---

### 1.4 Virtual Environments & Troubleshooting Mastered

1. **What is `.ps1`?**
   * `.ps1` stands for **PowerShell Script**. `Activate.ps1` is the script executed by Windows PowerShell to load the Python virtual environment (`.venv`).
2. **Resolving `ModuleNotFoundError: No module named 'google'`**
   * **Root Cause**: Running `uvicorn main:app --reload` using global Windows Python instead of the virtual environment.
   * **Fix**: Ensure terminal activation via `.\.venv\Scripts\Activate.ps1` before starting the server.

---

## Section 2: Detailed System Architecture & Mermaid Diagrams

This section provides visual architectural models of the entire system, data flow, and LLM interaction loops.

---

### 2.1 Overall System Architecture Diagram

```mermaid
flowchart TB
    subgraph ClientLayer["Client Layer"]
        User["User / Frontend Client"]
    end

    subgraph FastAPIBackend["FastAPI Application (main.py)"]
        API["FastAPI Server (Uvicorn)"]
        HealthEP["GET /health"]
        QueryEP["POST & GET /research-queries"]
        ChatEP["POST /chat"]
        
        subgraph ORMSection["Database Access Layer"]
            DB_Dep["get_db() Session Dependency"]
            ORM_Model["SQLAlchemy ORM (ResearchQuery)"]
        end
    end

    subgraph AgentEngine["Agentic Engine (agent.py)"]
        AgentFlow["run_agent_flow(user_message)"]
        GeminiClient["Google GenAI Client (gemini-3.6-flash)"]
        History["Message History Array"]
    end

    subgraph ToolLayer["Local Tool Definitions (tools.py)"]
        SearchTool["search_literature(query)"]
        PaperTool["get_paper(paper_id)"]
    end

    subgraph PersistenceLayer["Storage & External Services"]
        PostgreSQL[("PostgreSQL DB\n(neuro_research)")]
        GeminiAPI["Google Gemini API Cloud"]
    end

    %% Connections
    User -->|HTTP Requests| API
    API --> HealthEP
    API --> QueryEP
    API --> ChatEP
    
    QueryEP --> DB_Dep
    DB_Dep --> ORM_Model
    ORM_Model <-->|SQL Queries| PostgreSQL

    ChatEP --> AgentFlow
    AgentFlow --> GeminiClient
    GeminiClient <-->|HTTPS API Calls| GeminiAPI
    AgentFlow --> History
    AgentFlow -->|Executes Tool| ToolLayer
```

---

### 2.2 Sequence Diagram: Agent Tool Calling Lifecycle

The diagram below illustrates the exact sequence of events when a user asks a question requiring tool execution:

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Client
    participant API as FastAPI (/chat)
    participant Agent as agent.py (run_agent_flow)
    participant Gemini as Google Gemini API
    participant Tools as tools.py (Local Tools)

    User->>API: POST /chat {"message": "Find studies on eye-movement biomarkers"}
    API->>Agent: Invokes run_agent_flow(user_message)
    Agent->>Agent: Construct initial message history [Role: User]
    Agent->>Gemini: generate_content(tools=[search_literature, get_paper], auto_call=False)
    
    Note over Gemini: Gemini inspects query & tool signatures.<br/>Determines tool execution is required.
    
    Gemini-->>Agent: Returns FunctionCall Payload: search_literature(query="eye-movement biomarkers")
    
    Agent->>Agent: Append Model ToolCall response to Message History
    Agent->>Tools: Execute search_literature(query="eye-movement biomarkers")
    Tools-->>Agent: Return Python dict {"results": [paper1, paper2]}
    
    Agent->>Agent: Format result as Part.from_function_response()
    Agent->>Agent: Append FunctionResponse to Message History [Role: User]
    
    Agent->>Gemini: generate_content(updated message history with tool result)
    Gemini-->>Agent: Returns final synthesized natural language text
    Agent-->>API: Returns response string
    API-->>User: HTTP 200 OK {"response": "Based on recent literature..."}
```

---

### 2.3 Database Data Flow Diagram: Query Persistence

This diagram shows how a research query moves from raw JSON to PostgreSQL storage:

```mermaid
flowchart LR
    A["Raw JSON Input\n{'query': 'Alzheimer biomarkers'}"] 
    -->|Validation| B["Pydantic Model\nResearchQueryCreate"]
    -->|Instantiation| C["SQLAlchemy ORM Model\nResearchQuery(query=...)"]
    -->|db.add() & db.commit()| D[("PostgreSQL Database\nTable: research_queries")]
    -->|db.refresh()| E["Updated ORM Object\n(id=1, query=..., created_at=TIMESTAMP)"]
    -->|Serialization| F["HTTP Response JSON"]
```

---

## Section 3: Deep Dive into Codebase & Implementation Details

---

### 3.1 `main.py` — Database Models & API Endpoints

[`main.py`](file:///d:/Neuro%20Researcher/main.py) forms the primary entry point for our HTTP backend server.

#### Key Code Components:
1. **Database Session Setup**:
   ```python
   DATABASE_URL = "postgresql://postgres:arya21bhat@localhost:5432/neuro_research"
   engine = create_engine(DATABASE_URL)
   SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
   Base = declarative_base()

   def get_db():
       db = SessionLocal()
       try:
           yield db
       finally:
           db.close()
   ```

2. **SQLAlchemy ORM Table Definition**:
   ```python
   class ResearchQuery(Base):
       __tablename__ = "research_queries"

       id = Column(Integer, primary_key=True, index=True)
       query = Column(String, nullable=False)
       created_at = Column(DateTime(timezone=True), server_default=func.now())
   ```

3. **API Endpoints**:
   * `GET /health`: Health verification endpoint.
   * `POST /research-queries`: Saves research queries to database.
   * `GET /research-queries`: Lists all stored queries.
   * `GET /research-queries/{id}`: Fetches query by primary key ID.
   * `POST /chat`: Bridges user chat requests directly to `run_agent_flow()`.

---

### 3.2 `tools.py` — Domain Tool Definitions

[`tools.py`](file:///d:/Neuro%20Researcher/tools.py) contains local Python functions registered with Gemini.

1. `search_literature(query: str) -> dict`:
   * Accepts a scientific keyword search string.
   * Filters a mock scientific database containing papers on saccadic latency, anti-saccade tasks, and retinal imaging in cognitive decline.
   * Returns a structured dictionary `{"results": [...]}`.

2. `get_paper(paper_id: int) -> dict`:
   * Accepts an integer paper ID (e.g. `101`, `102`).
   * Looks up full paper abstract, authors, journal, and DOI.

---

### 3.3 `agent.py` — LLM Orchestration Logic

[`agent.py`](file:///d:/Neuro%20Researcher/agent.py) manages multi-turn communication with Gemini 3.6 Flash.

```python
def run_agent_flow(user_message: str) -> str:
    client = get_client()
    messages = [types.Content(role="user", parts=[types.Part.from_text(text=user_message)])]
    
    # 1. Initial Gemini call with manual tool execution
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=messages,
        config=types.GenerateContentConfig(
            tools=[search_literature, get_paper],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )
    )
    
    # 2. Check if model requested a tool call
    if response.function_calls:
        tool_call = response.function_calls[0]
        messages.append(response.candidates[0].content)
        
        # 3. Execute tool locally
        if tool_call.name == "search_literature":
            tool_output = search_literature(query=tool_call.args.get("query"))
        elif tool_call.name == "get_paper":
            tool_output = get_paper(paper_id=int(tool_call.args.get("paper_id")))
            
        # 4. Append tool response to message history
        tool_response_part = types.Part.from_function_response(
            name=tool_call.name,
            response=tool_output
        )
        messages.append(types.Content(role="user", parts=[tool_response_part]))
        
        # 5. Final synthesis call
        final_response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=messages,
            config=types.GenerateContentConfig(tools=[search_literature, get_paper])
        )
        return final_response.text

    return response.text
```

---

## Section 4: Project Directory Map

```
d:\Neuro Researcher\
├── .env                         # API Keys & Secrets
├── .gitignore                   # Version control rules
├── requirements.txt             # Installed dependencies
├── main.py                      # FastAPI Backend Server & Database Models
├── agent.py                     # Agentic Execution Loop
├── tools.py                     # Custom Local Tools (search_literature, get_paper)
├── test_llm.py                  # CLI Test Harness for Gemini Tool Calling
└── PROJECT_PROGRESS_SUMMARY.md # Complete Progress & Learning Document
```

---

## Section 5: Future Roadmap & Next Phases

* **Phase 3 — Real API Integrations**: Connect `tools.py` to live scientific APIs (PubMed / Semantic Scholar API).
* **Phase 4 — Conversation Persistence**: Store multi-turn user/agent chat histories in PostgreSQL tables.
* **Phase 5 — Research Assistant UI**: Build a modern web interface (Streamlit / React) for researchers to query biomarkers and literature.
