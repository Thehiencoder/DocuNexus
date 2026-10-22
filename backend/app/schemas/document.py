from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field, HttpUrl

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

class ScrapeRequest(BaseModel):
    """Payload for web URL scraping."""
    url: str = Field(..., description="Full HTTP/HTTPS URL of the web page to scrape.")
    title: Optional[str] = Field(None, description="Optional custom title for the document.")

class DocumentItem(BaseModel):
    """Catalog entry describing an ingested document."""
    doc_id: str
    filename: str
    source_type: str
    source_url: Optional[str] = None
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    total_pages: Optional[int] = None
    total_sections: int = 0
    total_chunks: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "indexed"

class DocumentUploadResponse(BaseModel):
    """Response returned upon successful file or URL ingestion."""
    doc_id: str
    filename: str
    source_type: str
    source_url: Optional[str] = None
    total_pages: Optional[int] = None
    total_sections: int
    total_chunks: int
    status: str = "success"
    message: str
    chunks_preview: List[DocumentChunk] = Field(default_factory=list)
