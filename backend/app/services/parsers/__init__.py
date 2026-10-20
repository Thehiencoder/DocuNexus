from app.services.parsers.dispatcher import parse_document, parse_url
from app.services.parsers.pdf_parser import parse_pdf
from app.services.parsers.docx_parser import parse_docx
from app.services.parsers.txt_parser import parse_txt
from app.services.parsers.web_parser import parse_web_page

__all__ = [
    "parse_document",
    "parse_url",
    "parse_pdf",
    "parse_docx",
    "parse_txt",
    "parse_web_page",
]
