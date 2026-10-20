import re
from pathlib import Path
from typing import Union, BinaryIO
from app.schemas.document import ParsedDocument, ParsedSection

def decode_bytes(content: bytes) -> str:
    """Attempt decoding using common text encodings."""
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")

def parse_txt(
    source: Union[str, Path, BinaryIO, bytes],
    doc_id: str,
    filename: str = "document.txt",
) -> ParsedDocument:
    """
    Parses plain text and Markdown files, segmenting by headers or paragraph blocks.
    """
    if isinstance(source, bytes):
        raw_content = decode_bytes(source)
    elif isinstance(source, (str, Path)) and Path(str(source)).exists():
        raw_content = decode_bytes(Path(str(source)).read_bytes())
    elif isinstance(source, str):
        raw_content = source
    elif hasattr(source, "read"):
        raw_content = decode_bytes(source.read())
    else:
        raw_content = str(source)

    sections = []
    lines = raw_content.splitlines()
    current_title = "Document Start"
    current_block = []

    def flush_block():
        nonlocal current_block, current_title
        if current_block:
            text = "\n".join(current_block).strip()
            if text:
                sections.append(
                    ParsedSection(
                        text=text,
                        section_title=current_title,
                        metadata={"lines": len(current_block)}
                    )
                )
            current_block = []

    for line in lines:
        stripped = line.strip()
        # Markdown heading check
        if re.match(r'^#{1,4}\s+(.+)$', stripped):
            flush_block()
            current_title = stripped.lstrip('#').strip()
        else:
            current_block.append(line)

    flush_block()

    if not sections and raw_content.strip():
        sections.append(
            ParsedSection(
                text=raw_content.strip(),
                section_title="Full Document",
                metadata={"length": len(raw_content)}
            )
        )

    return ParsedDocument(
        doc_id=doc_id,
        filename=filename,
        source_type="txt",
        source_url=None,
        sections=sections,
        total_pages=None,
        raw_text=raw_content,
        metadata={"char_count": len(raw_content), "section_count": len(sections)}
    )
