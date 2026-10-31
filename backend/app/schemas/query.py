from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class RetrievedChunk(BaseModel):
    """Represents a chunk retrieved by hybrid search with relevance score and metadata."""
    chunk_id: str
    doc_id: str
    filename: str
    source_type: str
    source_url: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    text: str
    score: float = 0.0
    char_start: int = 0
    char_end: int = 0
    chunk_index: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class HybridSearchRequest(BaseModel):
    """Payload for searching the knowledge base."""
    query: str = Field(..., description="User search query or question.")
    top_k: int = Field(25, description="Number of candidate chunks to retrieve from hybrid search.")
    doc_ids: Optional[List[str]] = Field(None, description="Optional document ID filters.")

class HybridSearchResponse(BaseModel):
    """Response containing retrieved candidate chunks."""
    query: str
    total_results: int
    results: List[RetrievedChunk]
