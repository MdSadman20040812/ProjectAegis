from typing import TypedDict, List, Optional, Any
from langchain_core.messages import BaseMessage

class AegisState(TypedDict):
    """State for the AEGIS cognitive tutor agent."""
    messages: List[BaseMessage]
    document_parsed: bool
    phase: Optional[str]
    segments: Optional[List[str]]
    current_segment_index: Optional[int]
    retriever: Optional[Any]
    difficulty: Optional[str]
