from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ParsedSection(BaseModel):
    """Represents a single structural section or page from a parsed document."""
    text: str = Field(..., description="Cleaned text content of the section or page.")
    page_number: Optional[int] = Field(None, description="Page number for paginated docs like PDFs (1-indexed).")
    section_title: Optional[str] = Field(None, description="Header, section title, or topic.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata for this section.")

class ParsedDocument(BaseModel):
    """Standardized representation of an ingested document across all formats."""
    doc_id: str = Field(..., description="Unique document ID (UUID).")
    filename: str = Field(..., description="Original filename or web page title.")
    source_type: str = Field(..., description="Format type: pdf, docx, txt, web.")
    source_url: Optional[str] = Field(None, description="Source URL if ingested from web.")
    sections: List[ParsedSection] = Field(default_factory=list, description="List of parsed pages or sections.")
    total_pages: Optional[int] = Field(None, description="Total pages if applicable.")
    raw_text: str = Field("", description="Full concatenated text.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional document metadata.")
