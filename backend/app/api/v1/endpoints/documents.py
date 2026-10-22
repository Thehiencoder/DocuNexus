import shutil
import uuid
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse
from app.schemas.document import (
    ScrapeRequest,
    DocumentItem,
    DocumentUploadResponse,
    DocumentChunk,
)
from app.services.parsers.dispatcher import parse_document, parse_url
from app.services.chunking import default_chunker
from app.services.catalog import catalog
from app.core.config import settings

router = APIRouter(prefix="/documents", tags=["Documents"])

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".csv", ".json"}

@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, DOCX, TXT, MD, etc.).
    Parses structural sections, generates granular chunks, and catalogs metadata.
    """
    filename = file.filename or "uploaded_document"
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    doc_id = f"doc_{uuid.uuid4().hex[:10]}"
    safe_name = f"{doc_id}_{filename}"
    saved_file_path = settings.upload_dir_path / safe_name

    # Save raw file locally
    try:
        with open(saved_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        )
    finally:
        file.file.close()

    file_size = saved_file_path.stat().st_size

    # Parse document structure
    try:
        parsed_doc = parse_document(source=saved_file_path, filename=filename, doc_id=doc_id)
    except Exception as e:
        saved_file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse document: {str(e)}",
        )

    # Chunk document into nodes
    try:
        nodes, chunks = default_chunker.chunk_document(parsed_doc)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed during chunking: {str(e)}",
        )

    # Save to catalog
    catalog_item = DocumentItem(
        doc_id=doc_id,
        filename=filename,
        source_type=parsed_doc.source_type,
        source_url=None,
        file_path=str(saved_file_path),
        file_size_bytes=file_size,
        total_pages=parsed_doc.total_pages,
        total_sections=len(parsed_doc.sections),
        total_chunks=len(chunks),
        status="indexed",
    )
    catalog.add_document(catalog_item)

    return DocumentUploadResponse(
        doc_id=doc_id,
        filename=filename,
        source_type=parsed_doc.source_type,
        source_url=None,
        total_pages=parsed_doc.total_pages,
        total_sections=len(parsed_doc.sections),
        total_chunks=len(chunks),
        status="success",
        message=f"Successfully parsed and chunked {len(chunks)} passages from '{filename}'.",
        chunks_preview=chunks[:3],  # Return first 3 chunks as preview
    )

@router.post("/scrape", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def scrape_web_page(payload: ScrapeRequest):
    """
    Ingests and extracts clean text from a public web page URL.
    """
    url_str = str(payload.url).strip()
    if not url_str.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL must start with http:// or https://",
        )

    doc_id = f"web_{uuid.uuid4().hex[:10]}"

    try:
        parsed_doc = parse_url(url=url_str, doc_id=doc_id, title=payload.title)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to scrape URL '{url_str}': {str(e)}",
        )

    # Save cached raw text for preview
    cached_file_path = settings.upload_dir_path / f"{doc_id}_web.txt"
    cached_file_path.write_text(parsed_doc.raw_text, encoding="utf-8")

    nodes, chunks = default_chunker.chunk_document(parsed_doc)

    catalog_item = DocumentItem(
        doc_id=doc_id,
        filename=parsed_doc.filename,
        source_type="web",
        source_url=url_str,
        file_path=str(cached_file_path),
        file_size_bytes=len(parsed_doc.raw_text.encode("utf-8")),
        total_pages=None,
        total_sections=len(parsed_doc.sections),
        total_chunks=len(chunks),
        status="indexed",
    )
    catalog.add_document(catalog_item)

    return DocumentUploadResponse(
        doc_id=doc_id,
        filename=parsed_doc.filename,
        source_type="web",
        source_url=url_str,
        total_pages=None,
        total_sections=len(parsed_doc.sections),
        total_chunks=len(chunks),
        status="success",
        message=f"Successfully scraped and chunked {len(chunks)} passages from '{url_str}'.",
        chunks_preview=chunks[:3],
    )

@router.get("", response_model=List[DocumentItem])
async def list_documents():
    """List all ingested documents in the knowledge base."""
    return catalog.list_documents()

@router.get("/{doc_id}", response_model=DocumentItem)
async def get_document(doc_id: str):
    """Retrieve metadata for a specific document."""
    doc = catalog.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc

@router.get("/{doc_id}/raw")
async def get_raw_document(doc_id: str):
    """Serve the raw original document file for the frontend viewer."""
    doc = catalog.get_document(doc_id)
    if not doc or not doc.file_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    
    file_path = Path(doc.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Original file not found on disk")

    media_type = "application/pdf" if doc.source_type == "pdf" else "text/plain"
    return FileResponse(
        path=file_path,
        filename=doc.filename,
        media_type=media_type,
    )

@router.delete("/{doc_id}", status_code=status.HTTP_200_OK)
async def delete_document(doc_id: str):
    """Remove a document from the catalog and purge local files."""
    doc = catalog.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    if doc.file_path:
        Path(doc.file_path).unlink(missing_ok=True)

    catalog.delete_document(doc_id)
    return {"status": "success", "message": f"Document '{doc.filename}' deleted successfully."}
