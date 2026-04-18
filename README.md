# AEGIS ⚡ — Cognitive Tutor

> **System-Directed Academic Breakdown Engine**
> Upload any academic document. Ask any question. AEGIS deconstructs it into first principles.

---

## What Is AEGIS?

AEGIS is an **Agentic RAG (Retrieval-Augmented Generation) system** built on LangGraph and Streamlit. It accepts complex academic source material (PDFs, lecture notes) and answers questions by breaking concepts down through a four-phase reasoning framework:

| Phase | Description |
|---|---|
| **A — Axiomatic Reduction** | Identifies absolute foundational truths from the document |
| **B — Reassembly** | Rebuilds the concept step-by-step (A → B → C) |
| **C — Simpler Terms** | Provides a universal 1:1 analogy |
| **D — Verification** | Asks a targeted question to confirm understanding |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Agentic Framework** | LangGraph (StateGraph + MemorySaver) |
| **LLM Provider** | Cerebras Inference (ultra-fast, dedicated hardware) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (local, no API cost) |
| **Vector Store** | ChromaDB (in-memory, per-session) |
| **UI** | Streamlit (premium dark glassmorphism theme) |
| **Document Loaders** | PyPDF + LangChain TextLoader |

---

## Setup

### 1. Prerequisites
- Python 3.10+
- A [Cerebras](https://cerebras.ai) API key

### 2. Clone & Install

```bash
git clone <your-repo-url>
cd Chatbot

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and fill in your key:

```bash
copy .env.example .env
```

Edit `.env`:
```
CEREBRAS_API_KEY=your_cerebras_api_key_here
```

Get your API key at: https://cerebras.ai

---

## Running AEGIS

**Option A — Double-click launcher (Windows):**
```
run.bat
```

**Option B — Manual:**
```bash
venv\Scripts\activate
streamlit run app.py
```

Then open your browser to: **http://localhost:8501**

---

## How to Use

1. **Launch** AEGIS via `run.bat` or the CLI command above
2. **Upload** your document (PDF or TXT) using the sidebar panel
3. **Ask** any question about the content in the chat input
4. **Adjust difficulty** on the fly:
   - Say `"simplify this"` or `"explain like I'm 5"` → removes jargon
   - Say `"go deeper"` or `"more technical"` → introduces formal math/nomenclature
5. **New Session** — click the reset button in the sidebar to start fresh
6. AEGIS will answer using only the uploaded document as context

---

## Architecture

```
app.py (Streamlit UI — premium dark glassmorphism theme)
│
├── agent/
│   ├── graph.py   — LangGraph StateGraph: conditional routing between init & query nodes
│   ├── nodes.py   — init_node (greeting) + process_query_node (RAG + Cerebras LLM reasoning)
│   ├── state.py   — AegisState TypedDict (messages, retriever, difficulty, etc.)
│   └── rag.py     — Document loader → chunker → HuggingFace embeddings → Chroma retriever
│
└── .env           — CEREBRAS_API_KEY
```

### LLM Fallback Chain (Cerebras Inference)
If any model returns an error, AEGIS automatically retries the next model:

1. `llama3.1-8b` (primary — fast)
2. `qwen-3-235b-a22b-instruct-2507`
3. `gpt-oss-120b`

---

## Project Structure

```
Chatbot/
├── agent/
│   ├── __init__.py
│   ├── graph.py
│   ├── nodes.py
│   ├── rag.py
│   └── state.py
├── .env               ← your API key (not committed)
├── .env.example       ← template
├── app.py             ← Streamlit entry point
├── requirements.txt
├── run.bat            ← Windows one-click launcher
└── README.md
```

---

## License

MIT — Built with LangGraph + Cerebras Inference.
