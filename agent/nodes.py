"""
Node implementations for AEGIS agent
Handles initialization and query processing with Cerebras LLM
"""

import os
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from agent.retriever_store import get_retriever

# Cerebras API configuration
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

# Model fallback chain
MODELS = [
    "llama3.1-8b",
    "qwen-3-235b-a22b-instruct-2507",
    "gpt-oss-120b"
]


def get_llm():
    """Get LLM with fallback chain."""
    for model in MODELS:
        try:
            llm = ChatOpenAI(
                model=model,
                api_key=CEREBRAS_API_KEY,
                base_url=CEREBRAS_BASE_URL,
                temperature=0.7,
            )
            return llm
        except Exception as e:
            print(f"Model {model} failed: {e}")
            continue
    raise RuntimeError("All LLM models failed")


def extract_assistant_output(content: str) -> str:
    """Extract clean assistant output from response."""
    # Remove any special markers or formatting
    if isinstance(content, str):
        # Clean up any XML-like tags or special markers
        content = content.strip()
    return content


def init_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize the agent with a greeting when no document is loaded.
    """
    greeting = (
        "⚡ **AEGIS Cognitive Tutor Initialized**\n\n"
        "I'm ready to help you understand academic material through first-principles reasoning.\n\n"
        "**To get started:**\n"
        "1. Upload a PDF or TXT document using the sidebar\n"
        "2. Ask any question about the content\n"
        "3. Say 'simplify this' or 'go deeper' to adjust difficulty\n\n"
        "What would you like to learn today?"
    )
    
    return {
        "messages": [AIMessage(content=greeting)],
        "phase": "WAITING_DOC"
    }


def process_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process user queries using RAG and Cerebras LLM.
    Implements the A-B-C-D reasoning framework:
    - A: Axiomatic Reduction
    - B: Reassembly
    - C: Simpler Terms
    - D: Verification
    """
    messages = state.get("messages", [])
    document_parsed = state.get("document_parsed", False)
    thread_id = state.get("thread_id", "default")
    
    if not messages:
        return {"messages": []}
    
    # Get the last user message
    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        return {"messages": messages}
    
    user_query = last_message.content
    
    # Check for difficulty adjustment
    difficulty = "standard"
    if "simplify" in user_query.lower() or "explain like i'm 5" in user_query.lower():
        difficulty = "simple"
    elif "go deeper" in user_query.lower() or "more technical" in user_query.lower():
        difficulty = "technical"
    
    # Get retriever if document is parsed
    retriever = None
    context = ""
    
    if document_parsed:
        retriever = get_retriever(thread_id)
        if retriever:
            try:
                docs = retriever.invoke(user_query)
                context = "\n\n".join([doc.content if hasattr(doc, 'content') else str(doc) for doc in docs])
            except Exception as e:
                print(f"Retrieval error: {e}")
    
    # Build system prompt based on A-B-C-D framework
    if document_parsed and context:
        system_prompt = f"""You are AEGIS, a Cognitive Tutor that explains concepts through first-principles reasoning.

Use the following framework to answer questions:

**A — Axiomatic Reduction**: Identify the foundational truths
**B — Reassembly**: Rebuild the concept step-by-step (A → B → C)
**C — Simpler Terms**: Provide a universal analogy
**D — Verification**: Ask a targeted question to confirm understanding

Context from document:
{context}

User's difficulty preference: {difficulty}

Answer using only the provided context. If the context doesn't contain the answer, say so clearly."""
    else:
        system_prompt = """You are AEGIS, a Cognitive Tutor. You help users understand concepts through first-principles reasoning.

Since no document has been uploaded yet, provide general explanations and encourage the user to upload material for more targeted assistance."""
    
    # Get LLM and generate response
    try:
        llm = get_llm()
        
        # Build message history
        chat_messages = [SystemMessage(content=system_prompt)]
        for msg in messages:
            chat_messages.append(msg)
        
        response = llm.invoke(chat_messages)
        
        return {
            "messages": messages + [response],
            "difficulty": difficulty
        }
    except Exception as e:
        error_message = f"I apologize, but I encountered an error processing your request: {str(e)}\n\nPlease try again or simplify your question."
        return {
            "messages": messages + [AIMessage(content=error_message)]
        }
