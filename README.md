# NeuroLens 🧠🔬

NeuroLens is an AI-powered scientific literature research assistant tailored for neurology. It leverages Google Gemini 3.6 Flash, Europe PMC REST APIs, and a FastAPI backend to fetch, summarize, and synthesize medical literature.

## Features
- **Biomedical Literature Search**: Query Europe PMC for real-time peer-reviewed literature.
- **Paper Detail Extraction**: Fetch complete abstracts, DOIs, authors, and publication info.
- **Agentic Function Calling**: Gemini model autonomously invokes tools to gather literature context before returning synthesized answers.
- **FastAPI & PostgreSQL Backend**: Persistent logging of research queries and chat interface endpoints.

## Quick Start

### 1. Setup Virtual Environment
```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Copy `.env.example` to `.env` and set your `GEMINI_API_KEY`:
```bash
cp .env.example .env
```

### 4. Run Application
```bash
uvicorn main:app --reload
```
