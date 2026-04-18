import streamlit as st
import os
import uuid
import tempfile
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

# Ensure the root directory is in the Python path so Streamlit Cloud can find 'agent'
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.graph import build_graph
from agent.rag import setup_rag_pipeline
from agent import retriever_store
from agent.nodes import extract_assistant_output

# ─────────────────────────────────────────────────────────────
# Environment & Page Config
# ─────────────────────────────────────────────────────────────
load_dotenv()

st.set_page_config(
    page_title="AEGIS | Cognitive Tutor",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Premium Dark Theme — Glassmorphism + Neon Accent System
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-primary: #08090d;
        --bg-secondary: #0e1117;
        --bg-card: rgba(17, 19, 28, 0.7);
        --bg-glass: rgba(255, 255, 255, 0.03);
        --border-subtle: rgba(255, 255, 255, 0.06);
        --border-accent: rgba(0, 255, 170, 0.15);
        --text-primary: #e8ecf1;
        --text-secondary: #8b95a5;
        --text-muted: #505868;
        --accent: #00ffaa;
        --accent-dim: rgba(0, 255, 170, 0.08);
        --accent-glow: rgba(0, 255, 170, 0.25);
        --accent-secondary: #6366f1;
        --danger: #ff4d6a;
        --success: #00d68f;
        --radius: 14px;
        --radius-sm: 8px;
    }

    /* ── Global ── */
    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Remove default streamlit padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 900px !important;
    }

    /* ── Header System ── */
    .aegis-header {
        text-align: center;
        padding: 2rem 0 1.5rem;
        position: relative;
    }

    .aegis-logo {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: 8px;
        text-transform: uppercase;
        background: linear-gradient(135deg, #00ffaa 0%, #6366f1 50%, #00ffaa 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient-shift 4s ease infinite;
        margin-bottom: 0.25rem;
    }

    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    .aegis-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        font-weight: 400;
        color: var(--text-muted);
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    .aegis-divider {
        width: 60px;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent), transparent);
        margin: 1.2rem auto 0;
        border-radius: 2px;
    }

    /* ── Status Chip ── */
    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 1rem;
    }

    .status-chip.active {
        background: rgba(0, 214, 143, 0.1);
        color: #00d68f;
        border: 1px solid rgba(0, 214, 143, 0.2);
    }

    .status-chip.waiting {
        background: rgba(255, 77, 106, 0.08);
        color: #ff4d6a;
        border: 1px solid rgba(255, 77, 106, 0.15);
    }

    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        animation: pulse-dot 2s ease infinite;
    }

    .status-chip.active .status-dot { background: #00d68f; }
    .status-chip.waiting .status-dot { background: #ff4d6a; }

    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent);
        font-size: 0.85rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] .stMarkdown p {
        color: var(--text-secondary);
        font-size: 0.82rem;
        line-height: 1.6;
    }

    /* ── File Uploader ── */
    section[data-testid="stFileUploader"] {
        border: 1px dashed var(--border-accent) !important;
        border-radius: var(--radius) !important;
        background: var(--accent-dim) !important;
        transition: all 0.3s ease;
    }

    section[data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
        background: rgba(0, 255, 170, 0.06) !important;
    }

    /* ── Chat Messages ── */
    .stChatMessage {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius) !important;
        padding: 1.2rem !important;
        margin-bottom: 1rem !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: border-color 0.3s ease;
    }

    .stChatMessage:hover {
        border-color: var(--border-accent) !important;
    }

    /* User message accent */
    .stChatMessage[data-testid="stChatMessage"]:has(.stAvatar:first-child) {
        border-left: 2px solid var(--accent-secondary) !important;
    }

    /* ── Chat Input ── */
    .stChatInput {
        border-radius: var(--radius) !important;
    }

    .stChatInput > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius) !important;
        transition: border-color 0.3s ease;
    }

    .stChatInput > div:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 20px var(--accent-dim) !important;
    }

    .stChatInput textarea {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, rgba(0, 255, 170, 0.1), rgba(99, 102, 241, 0.1)) !important;
        border: 1px solid var(--border-accent) !important;
        color: var(--accent) !important;
        border-radius: var(--radius-sm) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(0, 255, 170, 0.2), rgba(99, 102, 241, 0.2)) !important;
        border-color: var(--accent) !important;
        box-shadow: 0 4px 20px var(--accent-dim) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: var(--accent) !important;
    }

    /* ── Success/Info Alerts ── */
    .stAlert {
        border-radius: var(--radius-sm) !important;
        border: none !important;
        font-size: 0.85rem !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.08);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.15); }

    /* ── Markdown inside messages ── */
    .stChatMessage p {
        font-size: 0.92rem;
        line-height: 1.75;
        color: var(--text-primary);
    }

    .stChatMessage strong {
        color: var(--accent);
        font-weight: 700;
    }

    .stChatMessage code {
        background: rgba(99, 102, 241, 0.12);
        color: #a5b4fc;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
    }

    .stChatMessage hr {
        border-color: var(--border-subtle);
        margin: 1rem 0;
    }

    /* ── Sidebar Info Card ── */
    .sidebar-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        padding: 1rem;
        margin-top: 1.5rem;
    }

    .sidebar-card h4 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--text-muted);
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    .sidebar-card p {
        font-size: 0.78rem !important;
        color: var(--text-secondary) !important;
        margin: 0.3rem 0 !important;
    }

    .powered-by {
        font-size: 0.68rem;
        color: var(--text-muted);
        text-align: center;
        padding: 2rem 0 0.5rem;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="aegis-header">
    <div class="aegis-logo">AEGIS</div>
    <div class="aegis-subtitle">Cognitive Tutor System</div>
    <div class="aegis-divider"></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_parsed" not in st.session_state:
    st.session_state.document_parsed = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ─────────────────────────────────────────────────────────────
# Graph Config
# ─────────────────────────────────────────────────────────────
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ─────────────────────────────────────────────────────────────
# Sidebar — Context Ingestion Panel
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Context Ingestion")
    st.markdown("Upload source material to initialize the knowledge vector space.")

    uploaded_file = st.file_uploader(
        "Upload (PDF/TXT)",
        type=["pdf", "txt"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None and not st.session_state.document_parsed:
        extension = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        with st.spinner("⚡ Vectorizing document & identifying core concepts..."):
            retriever, full_text, segments = setup_rag_pipeline(tmp_path)
            st.session_state.document_parsed = True
            st.session_state.doc_name = uploaded_file.name
            
            # Store retriever in shared store (outside graph state)
            retriever_store.set_retriever(st.session_state.thread_id, retriever)

            # Invoking graph to kick off proposal node
            result = st.session_state.graph.invoke(
                {
                    "document_parsed": True, 
                    "phase": "WAITING_DOC", 
                    "segments": segments, 
                    "current_segment_index": 0
                },
                config=config,
            )
            st.session_state.messages = [msg for msg in result["messages"] if not isinstance(msg, SystemMessage)]
            st.rerun()

        st.success("Vector space initialized.")

    # Status indicator
    if st.session_state.document_parsed:
        st.markdown(f"""
        <div class="status-chip active">
            <span class="status-dot"></span> DOCUMENT LOADED
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"**File:** `{st.session_state.doc_name}`")
    else:
        st.markdown("""
        <div class="status-chip waiting">
            <span class="status-dot"></span> AWAITING UPLOAD
        </div>
        """, unsafe_allow_html=True)

    # Sidebar info card
    st.markdown("""
    <div class="sidebar-card">
        <h4>How to use</h4>
        <p>1. Upload a PDF or TXT document</p>
        <p>2. Ask any question about the content</p>
        <p>3. Say <strong>"simplify"</strong> or <strong>"go deeper"</strong> to adjust difficulty</p>
    </div>
    """, unsafe_allow_html=True)

    # New session button
    if st.button("🔄 New Session", use_container_width=True):
        for key in ["thread_id", "graph", "messages", "document_parsed", "retriever", "doc_name"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.markdown('<div class="powered-by">powered by cerebras inference</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Initialization Trigger (if graph has no memory yet)
# ─────────────────────────────────────────────────────────────
if not st.session_state.messages and not st.session_state.document_parsed:
    result = st.session_state.graph.invoke(
        {"messages": [], "document_parsed": False},
        config=config,
    )
    st.session_state.messages = [
        msg for msg in result["messages"] if not isinstance(msg, SystemMessage)
    ]

# ─────────────────────────────────────────────────────────────
# Render Conversation
# ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="⚡"):
            clean_content = extract_assistant_output(msg.content)
            st.markdown(clean_content)

# ─────────────────────────────────────────────────────────────
# Handle Input
# ─────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about your document..."):
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    human_msg = HumanMessage(content=prompt)

    with st.chat_message("assistant", avatar="⚡"):
        with st.spinner("Processing cognitive reduction..."):
            result = st.session_state.graph.invoke(
                {
                    "messages": [human_msg],
                    "document_parsed": st.session_state.document_parsed,
                },
                config=config,
            )

            clean_messages = [
                msg for msg in result["messages"] if not isinstance(msg, SystemMessage)
            ]
            st.session_state.messages = clean_messages

            if clean_messages and isinstance(clean_messages[-1], AIMessage):
                clean_content = extract_assistant_output(clean_messages[-1].content)
                st.markdown(clean_content)
