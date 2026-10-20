import uuid
from pathlib import Path
from typing import Union, BinaryIO, Optional
from app.schemas.document import ParsedDocument
from app.services.parsers.pdf_parser import parse_pdf
from app.services.parsers.docx_parser import parse_docx
from app.services.parsers.txt_parser import parse_txt
from app.services.parsers.web_parser import parse_web_page

def parse_document(
    source: Union[str, Path, BinaryIO, bytes],
    filename: str,
    doc_id: Optional[str] = None,
) -> ParsedDocument:
    """
    Auto-detects document format by file extension and executes the appropriate parser.
    """
    doc_id = doc_id or str(uuid.uuid4())
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return parse_pdf(source, doc_id=doc_id, filename=filename)
    elif ext in (".docx", ".doc"):
        return parse_docx(source, doc_id=doc_id, filename=filename)
    elif ext in (".txt", ".md", ".markdown", ".json", ".csv", ".log"):
        return parse_txt(source, doc_id=doc_id, filename=filename)
    else:
        # Fallback to text parser for unrecognized text-like files
        return parse_txt(source, doc_id=doc_id, filename=filename)

def parse_url(url: str, doc_id: Optional[str] = None, title: Optional[str] = None) -> ParsedDocument:
    """
    Parses a public web page URL.
    """
    doc_id = doc_id or str(uuid.uuid4())
    return parse_web_page(url=url, doc_id=doc_id, custom_title=title)
