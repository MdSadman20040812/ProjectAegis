"""
Retriever Store - manages retrievers outside of graph state
"""

_retrievers = {}


def set_retriever(thread_id: str, retriever) -> None:
    """Store a retriever for a given thread."""
    _retrievers[thread_id] = retriever


def get_retriever(thread_id: str):
    """Retrieve a retriever for a given thread."""
    return _retrievers.get(thread_id)


def clear_retriever(thread_id: str) -> None:
    """Clear a retriever for a given thread."""
    if thread_id in _retrievers:
        del _retrievers[thread_id]
