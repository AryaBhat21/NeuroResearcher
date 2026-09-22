"""
NeuroResearch — Interactive Streamlit Web Application.
Single-command UI for biomedical literature exploration & multi-turn clinical reasoning.
"""
import os
import json
import logging
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

# Import internal modules directly — with reload to ensure live code updates
import importlib
import agent
importlib.reload(agent)
from database import init_db, SessionLocal, check_db_connection, Conversation, Message
from agent import run_agent_turn, get_gemini_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neuro_research.ui")

# Page Configuration
st.set_page_config(
    page_title="NeuroResearch — Biomedical AI Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for a polished clinical UI
st.markdown("""
<style>
    .metric-badge {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        font-size: 0.82rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .badge-gemini {
        background-color: #1e3a8a;
        color: #bfdbfe;
        border: 1px solid #3b82f6;
    }
    .badge-deterministic {
        background-color: #312e81;
        color: #e0e7ff;
        border: 1px solid #6366f1;
    }
    .badge-postgres {
        background-color: #064e3b;
        color: #a7f3d0;
        border: 1px solid #10b981;
    }
    .badge-tool {
        background-color: #18181b;
        color: #f43f5e;
        border: 1px solid #71717a;
        font-family: monospace;
    }
    .paper-card {
        padding: 1rem;
        margin-bottom: 0.75rem;
        border-radius: 8px;
        background: #18181b;
        border: 1px solid #3f3f46;
    }
    .paper-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #60a5fa;
        text-decoration: none;
    }
    .paper-meta {
        font-size: 0.85rem;
        color: #d4d4d8;
        margin-top: 0.4rem;
        line-height: 1.5;
    }
    .suggestion-chip {
        font-size: 0.85rem;
        padding: 0.3rem 0.8rem;
        border-radius: 16px;
        background: #27272a;
        color: #e4e4e7;
        border: 1px solid #52525b;
        cursor: pointer;
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database tables if not already present
try:
    init_db()
except Exception as e:
    logger.warning(f"Database init exception: {e}")


def fetch_conversations():
    try:
        with SessionLocal() as db:
            return db.query(Conversation).order_by(Conversation.updated_at.desc()).all()
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        return []


def create_conversation(title: str = "New Research Session"):
    try:
        with SessionLocal() as db:
            conv = Conversation(title=title)
            db.add(conv)
            db.commit()
            db.refresh(conv)
            return conv.id, conv.title
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        return None, title


def load_conversation_messages(conv_id: int):
    try:
        with SessionLocal() as db:
            msgs = db.query(Message).filter(Message.conversation_id == conv_id).order_by(Message.id.asc()).all()
            result = []
            for m in msgs:
                tools = []
                sources = []
                if m.tool_calls:
                    try:
                        parsed = json.loads(m.tool_calls)
                        if isinstance(parsed, dict):
                            tools = parsed.get("tools_used", [])
                        elif isinstance(parsed, list):
                            tools = [t.get("tool", "") for t in parsed if isinstance(t, dict)]
                    except Exception:
                        pass
                result.append({
                    "role": m.role,
                    "content": m.content,
                    "tools": tools,
                    "sources": sources
                })
            return result
    except Exception as e:
        logger.error(f"Error loading messages: {e}")
        return []


# Session State Initialization
if "current_conv_id" not in st.session_state:
    st.session_state.current_conv_id = None
if "current_conv_title" not in st.session_state:
    st.session_state.current_conv_title = "New Research Session"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "preset_prompt" not in st.session_state:
    st.session_state.preset_prompt = None

# If there's an active conversation, keep messages synced
if st.session_state.current_conv_id and not st.session_state.messages:
    st.session_state.messages = load_conversation_messages(st.session_state.current_conv_id)


# Sidebar Configuration
with st.sidebar:
    st.title("🧠 NeuroResearch")
    st.caption("Biomedical Literature & Multi-Turn Clinical Reasoning Agent")
    st.divider()

    # System Status Section
    st.subheader("System Status")
    db_ok, db_msg = check_db_connection()
    gemini_client = get_gemini_client()
    gemini_live = gemini_client is not None
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    if db_ok:
        st.markdown('<span class="metric-badge badge-postgres">● PostgreSQL: Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="metric-badge" style="background:#7f1d1d;color:#fca5a5;">● PostgreSQL: {db_msg}</span>', unsafe_allow_html=True)

    if gemini_live:
        st.markdown(f'<span class="metric-badge badge-gemini">● Engine: Google {gemini_model}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="metric-badge badge-deterministic">● Engine: Deterministic Clinical Fallback</span>', unsafe_allow_html=True)

    st.markdown('<span class="metric-badge badge-tool">● Europe PMC REST: Live</span>', unsafe_allow_html=True)

    # API Key & Reasoning Engine Settings Drawer
    with st.expander("⚙️ AI Reasoning Engine Settings", expanded=False):
        st.caption("NeuroResearch uses a dual-engine architecture. Paste a Google Gemini API key to activate full generative dialogue, or use the offline deterministic engine.")
        api_input = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            if st.button("Apply Key", use_container_width=True):
                if api_input.strip():
                    try:
                        from google import genai
                        test_client = genai.Client(api_key=api_input.strip())
                        test_res = test_client.models.generate_content(model="gemini-2.5-flash", contents="Hi")
                        if test_res.text:
                            os.environ["GEMINI_API_KEY"] = api_input.strip()
                            import agent
                            agent._gemini_available = True
                            st.success("Valid Gemini Key! Switched to Live Gemini Engine.")
                            st.rerun()
                    except Exception as err:
                        st.error(f"Invalid API Key: {str(err)[:80]}")
        with col_k2:
            if st.button("Use Offline", use_container_width=True):
                os.environ["GEMINI_API_KEY"] = ""
                import agent
                agent._gemini_available = False
                st.info("Switched to Deterministic Clinical Engine.")
                st.rerun()

    st.divider()

    # New Conversation Button
    if st.button("➕ New Research Session", use_container_width=True, type="primary"):
        cid, title = create_conversation(title="New Research Session")
        st.session_state.current_conv_id = cid
        st.session_state.current_conv_title = title
        st.session_state.messages = []
        st.rerun()

    # Saved Conversations List
    st.subheader("Research Sessions")
    convs = fetch_conversations()
    if convs:
        for c in convs:
            active_mark = "👉 " if c.id == st.session_state.current_conv_id else ""
            label = f"{active_mark}{c.title[:28]}..." if len(c.title) > 28 else f"{active_mark}{c.title}"
            if st.button(label, key=f"conv_{c.id}", use_container_width=True):
                st.session_state.current_conv_id = c.id
                st.session_state.current_conv_title = c.title
                st.session_state.messages = load_conversation_messages(c.id)
                st.rerun()
    else:
        st.caption("No previous sessions found.")

    st.divider()

    # Quick Example Prompts
    st.subheader("Sample Prompts")
    sample_queries = [
        "What is neuroplasticity?",
        "Find recent studies on saccadic eye movements in Alzheimer disease",
        "How is it different from Parkinson disease?",
        "What biomarkers correlate with ALS oculomotor deficits?",
        "Retrieve metadata for paper PMID: 38192014"
    ]
    for sq in sample_queries:
        if st.button(sq, key=f"sq_{hash(sq)}", use_container_width=True):
            st.session_state.preset_prompt = sq
            st.rerun()


# Main View Header
header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    session_title = st.session_state.current_conv_title
    cid_str = f" (ID: #{st.session_state.current_conv_id})" if st.session_state.current_conv_id else ""
    st.subheader(f"💬 {session_title}{cid_str}")
with header_col2:
    if st.session_state.messages:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            cid, title = create_conversation(title="New Research Session")
            st.session_state.current_conv_id = cid
            st.session_state.current_conv_title = title
            st.rerun()

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍⚕️" if msg["role"] == "user" else "🧠"):
        st.markdown(msg["content"])

        # Display Tool badges if used
        if msg.get("tools"):
            tool_badges = " ".join([f"`{t}`" for t in msg["tools"]])
            st.markdown(f"**Tools Executed:** {tool_badges}")

        # Display verified citation cards
        sources = msg.get("sources", [])
        if sources:
            with st.expander(f"📚 Verified Sources & Citations ({len(sources)})", expanded=False):
                for idx, src in enumerate(sources, 1):
                    title = src.get("title", "Untitled Paper")
                    authors = ", ".join(src.get("authors", [])) or "Authors not listed"
                    year = src.get("publication_year", "N/A")
                    journal = src.get("journal", "Journal not specified")
                    pmid = src.get("id", "")
                    doi = src.get("doi", "")
                    url = src.get("url", f"https://europepmc.org/article/MED/{pmid}" if pmid else "#")

                    st.markdown(f"""
                    <div class="paper-card">
                        <a href="{url}" target="_blank" class="paper-title">#{idx}. {title}</a>
                        <div class="paper-meta">
                            <strong>Authors:</strong> {authors}<br>
                            <strong>Journal:</strong> {journal} ({year}) | <strong>PMID:</strong> `{pmid}` {f"| <strong>DOI:</strong> `{doi}`" if doi else ""}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# Show helpful follow-up suggestions if there are messages
if st.session_state.messages:
    st.caption("Suggested Follow-ups:")
    sug_col1, sug_col2, sug_col3 = st.columns(3)
    with sug_col1:
        if st.button("📄 Fetch paper ID with high citation", key="sug_1", use_container_width=True):
            st.session_state.preset_prompt = "fetch the paper id with high citation"
            st.rerun()
    with sug_col2:
        if st.button("🔬 What cellular mechanisms drive this?", key="sug_2", use_container_width=True):
            st.session_state.preset_prompt = "what cellular and molecular mechanisms drive this?"
            st.rerun()
    with sug_col3:
        if st.button("⚖️ Compare with Parkinson's disease", key="sug_3", use_container_width=True):
            st.session_state.preset_prompt = "How does this compare with Parkinson's disease?"
            st.rerun()


# Handle user input via either preset prompt or regular chat input
prompt_from_input = st.chat_input("Ask a clinical research question (e.g., 'What is neuroplasticity?', 'Find studies on saccadic latency in ALS')...")
user_prompt = st.session_state.preset_prompt or prompt_from_input

if user_prompt:
    st.session_state.preset_prompt = None

    # If no active conversation, create one now in DB
    if not st.session_state.current_conv_id:
        title_summary = user_prompt[:40] + "..." if len(user_prompt) > 40 else user_prompt
        cid, title = create_conversation(title=title_summary)
        st.session_state.current_conv_id = cid
        st.session_state.current_conv_title = title

    # Append user message to UI state immediately
    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt,
        "tools": [],
        "sources": []
    })

    # Render user turn
    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(user_prompt)

    # Execute Agent turn with Spinner
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Analyzing clinical inquiry & retrieving Europe PMC literature..."):
            with SessionLocal() as db:
                agent_result = run_agent_turn(
                    db=db,
                    conversation_id=st.session_state.current_conv_id,
                    user_message=user_prompt
                )

                response_text = agent_result.response
                tools_used = agent_result.tools_used
                sources = agent_result.sources
                st.session_state.current_conv_id = agent_result.conversation_id

                # Update conversation title if still default
                conv_obj = db.query(Conversation).filter(Conversation.id == st.session_state.current_conv_id).first()
                if conv_obj and conv_obj.title in ["New Research Session", "New Conversation"]:
                    new_title = user_prompt[:35] + "..." if len(user_prompt) > 35 else user_prompt
                    conv_obj.title = new_title
                    conv_obj.updated_at = datetime.utcnow()
                    db.commit()
                    st.session_state.current_conv_title = new_title

        # Display assistant response
        st.markdown(response_text)

        if tools_used:
            tool_badges = " ".join([f"`{t}`" for t in tools_used])
            st.markdown(f"**Tools Executed:** {tool_badges}")

        if sources:
            with st.expander(f"📚 Verified Sources & Citations ({len(sources)})", expanded=True):
                for idx, src in enumerate(sources, 1):
                    title = src.get("title", "Untitled Paper")
                    authors = ", ".join(src.get("authors", [])) or "Authors not listed"
                    year = src.get("publication_year", "N/A")
                    journal = src.get("journal", "Journal not specified")
                    pmid = src.get("id", "")
                    doi = src.get("doi", "")
                    url = src.get("url", f"https://europepmc.org/article/MED/{pmid}" if pmid else "#")

                    st.markdown(f"""
                    <div class="paper-card">
                        <a href="{url}" target="_blank" class="paper-title">#{idx}. {title}</a>
                        <div class="paper-meta">
                            <strong>Authors:</strong> {authors}<br>
                            <strong>Journal:</strong> {journal} ({year}) | <strong>PMID:</strong> `{pmid}` {f"| <strong>DOI:</strong> `{doi}`" if doi else ""}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Append to UI session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "tools": tools_used,
            "sources": sources
        })
        st.rerun()
