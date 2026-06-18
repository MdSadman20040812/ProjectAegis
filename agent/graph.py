"""
LangGraph StateGraph for AEGIS Cognitive Tutor
Implements conditional routing between initialization and query processing nodes.
"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from agent.state import AegisState
from agent.nodes import init_node, process_query_node


def should_process_query(state: Dict[str, Any]) -> Literal["init_node", "process_query_node"]:
    """
    Determine which node to route to based on state.
    """
    messages = state.get("messages", [])
    document_parsed = state.get("document_parsed", False)
    
    # If no messages, initialize
    if not messages:
        return "init_node"
    
    # If document is parsed and we have messages, process query
    if document_parsed:
        return "process_query_node"
    
    # Default to init for greeting
    return "init_node"


def build_graph():
    """
    Build and compile the AEGIS StateGraph.
    """
    # Create the graph builder
    builder = StateGraph(AegisState)
    
    # Add nodes
    builder.add_node("init_node", init_node)
    builder.add_node("process_query_node", process_query_node)
    
    # Add edges
    builder.add_edge(START, "init_node")
    builder.add_conditional_edges(
        "init_node",
        should_process_query,
        {
            "init_node": "init_node",
            "process_query_node": "process_query_node",
        }
    )
    
    # Query processing always ends
    builder.add_edge("process_query_node", END)
    
    # Compile with memory saver for checkpointing
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    return graph
