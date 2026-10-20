import io
import re
from pathlib import Path
from typing import Union, BinaryIO
from pypdf import PdfReader
from app.schemas.document import ParsedDocument, ParsedSection

def clean_extracted_text(text: str) -> str:
    """Normalize whitespace and remove excessive blank lines."""
    if not text:
        return ""
    # Normalize multiple newlines to double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove excessive horizontal spaces
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def parse_pdf(
    source: Union[str, Path, BinaryIO, bytes],
    doc_id: str,
    filename: str = "document.pdf",
) -> ParsedDocument:
    """
    Extracts text from a PDF document, preserving exact page numbers for citation.
    """
    if isinstance(source, bytes):
        reader = PdfReader(io.BytesIO(source))
    elif isinstance(source, (str, Path)):
        reader = PdfReader(str(source))
    else:
        reader = PdfReader(source)

    sections = []
    full_text_parts = []
    total_pages = len(reader.pages)

    for idx, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as e:
            page_text = f"[Error extracting text from page {idx}: {str(e)}]"

        cleaned = clean_extracted_text(page_text)
        if cleaned:
            sections.append(
                ParsedSection(
                    text=cleaned,
                    page_number=idx,
                    section_title=f"Page {idx}",
                    metadata={"page_index": idx - 1, "char_count": len(cleaned)}
                )
            )
            full_text_parts.append(f"--- Page {idx} ---\n{cleaned}")

    return ParsedDocument(
        doc_id=doc_id,
        filename=filename,
        source_type="pdf",
        source_url=None,
        sections=sections,
        total_pages=total_pages,
        raw_text="\n\n".join(full_text_parts),
        metadata={"total_pages": total_pages, "section_count": len(sections)}
    )
