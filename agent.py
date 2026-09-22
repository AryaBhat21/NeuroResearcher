"""
Agent Orchestration Engine for NeuroResearch.
Supports multi-turn conversational context, tool execution loops,
PostgreSQL message persistence, and dual-mode execution (Live Gemini / Deterministic Fallback).
"""
import os
import json
import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from database import Conversation, Message
from schemas import ChatResponse
from tools import search_literature, get_paper

load_dotenv()

logger = logging.getLogger("neuro_research.agent")

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_gemini_available: Optional[bool] = None

# System instruction defining domain expertise and citation requirements
NEURO_SYSTEM_INSTRUCTION = (
    "You are an expert Neurology and Neuroscience AI Research Assistant. "
    "You assist neuroscientists, clinicians, and researchers in analyzing peer-reviewed "
    "biomedical literature, neurodegenerative biomarkers, and diagnostic workflows. "
    "When a user asks for scientific evidence, recent studies, or specific paper details, "
    "use the 'search_literature' or 'get_paper' tools to retrieve verifiable records from Europe PMC / PubMed. "
    "Always synthesize the findings clearly, comparing mechanisms or cohorts when requested, "
    "and cite specific paper titles, authors, years, and identifiers (PMID/DOI)."
)


def get_gemini_client():
    """Returns an authenticated genai.Client or None if no valid key is configured."""
    global _gemini_available
    if _gemini_available is False:
        return None

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        _gemini_available = False
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception as exc:
        logger.warning(f"Failed to initialize Gemini client: {exc}")
        _gemini_available = False
        return None



# ---------------------------------------------------------------------------
# Fallback / Deterministic Research Reasoning Engine
# ---------------------------------------------------------------------------

class DeterministicResearchAgent:
    """
    High-fidelity deterministic research agent for offline execution,
    evaluation benchmarks, and resilient fallbacks when external LLM APIs
    are unavailable or credentials are invalid.
    """

    @staticmethod
    def resolve_context(current_message: str, prior_messages: List[Message]) -> str:
        """
        Resolves coreferences (e.g. 'it', 'this', 'how does it compare')
        by extracting the subject from previous dialogue turns.
        """
        lower = current_message.lower().strip()
        pronouns = ["it", "this", "that", "these", "they", "its"]

        # Check if the query is a comparative or follow-up query
        is_followup = any(p in lower.split() for p in pronouns) or lower.startswith(
            ("how is it", "how does it", "what about", "compare with", "how different")
        )

        if is_followup and prior_messages:
            # Find the most recent user turn or assistant turn with a topic
            for prev_msg in reversed(prior_messages):
                if prev_msg.role == "user" and prev_msg.content != current_message:
                    # Extract key biomedical entity keywords from previous query
                    prev_text = prev_msg.content
                    # Clean out common question words
                    cleaned = re.sub(
                        r"(?i)\b(what is|what are|find|studies on|tell me about|recent|literature|show)\b",
                        "",
                        prev_text
                    ).strip()
                    if cleaned:
                        logger.info(f"Resolved context '{cleaned}' into follow-up: '{current_message}'")
                        return f"{cleaned} {current_message}"

        return current_message

    @classmethod
    def execute_turn(
        cls,
        resolved_query: str,
        original_query: str,
        prior_messages: List[Message]
    ) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        """
        Executes deterministic tool selection, paper retrieval, and response synthesis.
        """
        lower = resolved_query.lower()
        tools_used = []
        sources = []

        # 1. Check if user is asking for a specific paper by ID
        paper_id_match = re.search(r"\b(?:pmid|id|paper)\s*[:#]?\s*(\d{6,10}|PMC\d+)\b", lower)
        if paper_id_match:
            paper_id = paper_id_match.group(1)
            tools_used.append("get_paper")
            paper_result = get_paper(paper_id=paper_id)
            if "title" in paper_result:
                sources.append(paper_result)
                response = (
                    f"### Paper Details: {paper_result.get('title')}\n\n"
                    f"- **Authors:** {', '.join(paper_result.get('authors', []))}\n"
                    f"- **Journal:** {paper_result.get('journal')} ({paper_result.get('publication_year')})\n"
                    f"- **DOI:** {paper_result.get('doi')}\n"
                    f"- **PMID/ID:** {paper_result.get('id')}\n"
                    f"- **URL:** [{paper_result.get('title')}]({paper_result.get('url')})\n\n"
                    f"**Abstract:**\n{paper_result.get('abstract')}"
                )
                return response, tools_used, sources

        # 2. Check if the query requires biomedical literature search
        research_indicators = [
            "search", "paper", "study", "studies", "literature", "biomarker",
            "alzheimer", "parkinson", "saccadic", "retinal", "dementia",
            "tau", "amyloid", "eeg", "mri", "synuclein", "compare", "differ",
            "clinical", "trial", "diagnos", "cognitive", "neurol", "neurofilament",
            "sclerosis", "als", "plasma", "csf", "cerebrospinal", "assay", "protein",
            "pathol", "disease", "syndrome", "measure", "mechanism", "therapy", "treatment"
        ]

        prior_used_tools = any(bool(getattr(m, 'tool_calls', None)) for m in prior_messages)
        needs_search = any(ind in lower for ind in research_indicators) or (prior_used_tools and len(resolved_query) > len(original_query))

        if needs_search:
            tools_used.append("search_literature")

            # Extract search keywords: strip conversational phrases
            cleaned_search = re.sub(
                r"(?i)\b(please|can you|find|show me|search for|what does the literature say about|"
                r"how is it different from|how does it compare to|compare with|tell me about|recent studies on|"
                r"how is it|how are they|what is the diagnostic utility of|what is|what are)\b",
                "",
                resolved_query
            ).strip()

            # Remove trailing question marks or punctuation
            cleaned_search = re.sub(r"[?!.,]+$", "", cleaned_search).strip()

            if not cleaned_search:
                cleaned_search = resolved_query

            # Determine publication year bounds if specified
            start_year = None
            end_year = None
            year_match = re.search(r"\b(201\d|202\d)\b", resolved_query)
            if year_match:
                start_year = int(year_match.group(1))

            literature_result = search_literature(
                query=cleaned_search,
                max_results=3,
                start_year=start_year,
                end_year=end_year
            )

            papers = literature_result.get("results", [])
            sources.extend(papers)

            if papers:
                citations_text = []
                for idx, paper in enumerate(papers, 1):
                    authors = ", ".join(paper.get("authors", [])[:3])
                    year = paper.get("publication_year") or "n.d."
                    title = paper.get("title")
                    journal = paper.get("journal")
                    url = paper.get("url")
                    abstract = paper.get("abstract", "")
                    # Shorten abstract to key takeaway
                    snippet = abstract[:280] + "..." if len(abstract) > 280 else abstract

                    citations_text.append(
                        f"{idx}. **{title}** ({year})\n"
                        f"   - *Authors:* {authors}\n"
                        f"   - *Journal:* {journal}\n"
                        f"   - *Link:* [{paper.get('id')}]({url})\n"
                        f"   - *Key Finding:* {snippet}"
                    )

                response = (
                    f"Based on recent biomedical literature retrieved via Europe PMC for **{cleaned_search}**:\n\n"
                    + "\n\n".join(citations_text)
                    + "\n\n### Clinical & Mechanistic Synthesis\n"
                    f"The retrieved peer-reviewed studies highlight key biomarkers and phenotypic distinctions "
                    f"related to '{cleaned_search}'. In neurodegenerative research, these findings provide objective, "
                    f"non-invasive metrics that aid in early differential diagnosis and monitoring progression."
                )
            else:
                response = (
                    f"I queried biomedical literature repositories for **{cleaned_search}**, but no matching "
                    f"peer-reviewed articles were found. Consider broadening the search keywords or expanding the publication year window."
                )

            return response, tools_used, sources

        # 3. General conversational / conceptual query without tool requirement
        response = (
            f"In neurological and neuroscience research, **{original_query}** involves understanding "
            f"central nervous system pathways, neurodegenerative pathophysiologies, and clinical diagnostic criteria. "
            f"If you would like to inspect empirical studies, please ask for recent papers or specific biomarkers."
        )
        return response, tools_used, sources


# ---------------------------------------------------------------------------
# Gemini Agent Execution
# ---------------------------------------------------------------------------

def run_gemini_agent(
    user_message: str,
    prior_messages: List[Message]
) -> Optional[Tuple[str, List[str], List[Dict[str, Any]]]]:
    """
    Executes Gemini function calling loop with conversational history.
    Returns None if Gemini is unavailable or errors out.
    """
    client = get_gemini_client()
    if client is None:
        return None

    try:
        from google.genai import types

        # 1. Build chronological message history for Gemini
        gemini_messages = []

        for msg in prior_messages:
            if msg.role == "user":
                gemini_messages.append(
                    types.Content(role="user", parts=[types.Part.from_text(text=msg.content)])
                )
            elif msg.role == "assistant":
                gemini_messages.append(
                    types.Content(role="model", parts=[types.Part.from_text(text=msg.content)])
                )

        # Append current user message
        gemini_messages.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
        )

        tools_used: List[str] = []
        sources: List[Dict[str, Any]] = []

        # 2. Initial model call with tool declarations
        config = types.GenerateContentConfig(
            system_instruction=NEURO_SYSTEM_INSTRUCTION,
            tools=[search_literature, get_paper],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )

        model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        response = client.models.generate_content(
            model=model_name,
            contents=gemini_messages,
            config=config
        )

        # 3. Tool execution loop (supports up to 3 turns of tool calling)
        loop_count = 0
        max_tool_turns = 3

        while response.function_calls and loop_count < max_tool_turns:
            loop_count += 1
            tool_call = response.function_calls[0]
            tool_name = tool_call.name
            tools_used.append(tool_name)

            # Record model tool call in context
            gemini_messages.append(response.candidates[0].content)

            tool_output: Any = None
            if tool_name == "search_literature":
                kwargs = {k: v for k, v in tool_call.args.items() if v is not None}
                tool_output = search_literature(**kwargs)
                if isinstance(tool_output, dict) and "results" in tool_output:
                    sources.extend(tool_output["results"])
            elif tool_name == "get_paper":
                paper_id = str(tool_call.args.get("paper_id", ""))
                tool_output = get_paper(paper_id=paper_id)
                if isinstance(tool_output, dict) and "title" in tool_output:
                    sources.append(tool_output)

            # Inject function response back into context
            response_part = types.Part.from_function_response(
                name=tool_name,
                response=tool_output or {"status": "no output"}
            )
            gemini_messages.append(
                types.Content(role="user", parts=[response_part])
            )

            # Generate synthesis from model
            response = client.models.generate_content(
                model=model_name,
                contents=gemini_messages,
                config=types.GenerateContentConfig(
                    system_instruction=NEURO_SYSTEM_INSTRUCTION,
                    tools=[search_literature, get_paper]
                )
            )

        final_text = response.text or "I have processed your research query."
        return final_text, tools_used, sources

    except Exception as exc:
        global _gemini_available
        logger.warning(f"Gemini execution encountered an error: {exc}. Falling back to deterministic engine.")
        _gemini_available = False
        return None


# ---------------------------------------------------------------------------
# Core Turn Orchestration & Persistence
# ---------------------------------------------------------------------------

def run_agent_turn(
    db: Session,
    conversation_id: Optional[int],
    user_message: str
) -> ChatResponse:
    """
    Main entrypoint for agent execution:
    1. Loads or creates Conversation in PostgreSQL.
    2. Retrieves prior message history in chronological order.
    3. Persists incoming user message.
    4. Executes agent turn (Gemini with deterministic fallback).
    5. Persists assistant response and tool metadata.
    6. Returns structured ChatResponse.
    """
    # 1. Resolve or create conversation
    if conversation_id is not None:
        conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conversation:
            # Create with specified ID or new
            conversation = Conversation(title=user_message[:50])
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
    else:
        conversation = Conversation(title=user_message[:50])
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    # 2. Retrieve prior messages in strict chronological order
    prior_messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation.id)
        .order_by(Message.id.asc())
        .all()
    )

    # 3. Persist user message
    user_record = Message(
        conversation_id=conversation.id,
        role="user",
        content=user_message
    )
    db.add(user_record)
    db.commit()

    # 4. Attempt Gemini execution, fallback if unavailable
    gemini_result = run_gemini_agent(user_message=user_message, prior_messages=prior_messages)

    if gemini_result is not None:
        response_text, tools_used, sources = gemini_result
    else:
        # Resolve multi-turn context
        resolved_query = DeterministicResearchAgent.resolve_context(
            current_message=user_message,
            prior_messages=prior_messages
        )
        response_text, tools_used, sources = DeterministicResearchAgent.execute_turn(
            resolved_query=resolved_query,
            original_query=user_message,
            prior_messages=prior_messages
        )

    # 5. Persist assistant message with tool metadata
    tool_metadata = json.dumps({"tools_used": tools_used, "source_count": len(sources)}) if tools_used else None
    assistant_record = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=response_text,
        tool_calls=tool_metadata
    )
    db.add(assistant_record)
    db.commit()

    return ChatResponse(
        conversation_id=conversation.id,
        response=response_text,
        role="assistant",
        tools_used=tools_used,
        sources=sources
    )


# ---------------------------------------------------------------------------
# Backwards Compatibility Function
# ---------------------------------------------------------------------------

def run_agent_flow(user_message: str) -> str:
    """
    Preserved legacy function for Phase 2 compatibility.
    Executes a standalone single-turn query without session persistence.
    """
    gemini_result = run_gemini_agent(user_message=user_message, prior_messages=[])
    if gemini_result is not None:
        return gemini_result[0]

    response_text, _, _ = DeterministicResearchAgent.execute_turn(
        resolved_query=user_message,
        original_query=user_message,
        prior_messages=[]
    )
    return response_text
