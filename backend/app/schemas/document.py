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

class DocumentChunk(BaseModel):
    """A granular chunk/node ready for embedding, vector storage, and citation attribution."""
    chunk_id: str = Field(..., description="Unique chunk UUID.")
    doc_id: str = Field(..., description="Parent document UUID.")
    filename: str = Field(..., description="Document filename.")
    source_type: str = Field(..., description="Source type: pdf, docx, txt, web.")
    source_url: Optional[str] = Field(None, description="Source URL for web documents.")
    page_number: Optional[int] = Field(None, description="1-indexed page number (crucial for PDFs).")
    section_title: Optional[str] = Field(None, description="Associated section or chapter title.")
    text: str = Field(..., description="Text content of the chunk.")
    char_start: int = Field(0, description="Start character offset within section/document.")
    char_end: int = Field(0, description="End character offset within section/document.")
    chunk_index: int = Field(0, description="Sequential index of the chunk within the document.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata payload.")
