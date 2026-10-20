import re
from typing import Optional
import httpx
import trafilatura
from bs4 import BeautifulSoup
from app.schemas.document import ParsedDocument, ParsedSection

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

def parse_web_page(
    url: str,
    doc_id: str,
    custom_title: Optional[str] = None,
    timeout_sec: float = 15.0,
) -> ParsedDocument:
    """
    Downloads and extracts clean article text from a public web page,
    stripping ads, navigations, and footers via trafilatura.
    """
    downloaded = None
    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception:
        downloaded = None

    # Fallback to httpx if trafilatura fetch fails
    if not downloaded:
        with httpx.Client(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=timeout_sec) as client:
            resp = client.get(url)
            resp.raise_for_status()
            downloaded = resp.text

    # Extract structured article text
    extracted_text = trafilatura.extract(
        downloaded,
        include_links=False,
        include_images=False,
        include_tables=True,
        output_format="txt",
    )

    metadata = trafilatura.extract_metadata(downloaded) if downloaded else None
    title = custom_title or (metadata.title if metadata and metadata.title else None)

    # Fallback with BeautifulSoup if trafilatura extraction produced nothing
    if not extracted_text:
        soup = BeautifulSoup(downloaded or "", "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()
        if not title and soup.title:
            title = soup.title.string
        extracted_text = soup.get_text(separator="\n\n").strip()

    title = title or url
    cleaned_text = re.sub(r'\n{3,}', '\n\n', extracted_text or "").strip()

    # Split into sections by paragraphs or headings
    raw_blocks = [b.strip() for b in cleaned_text.split("\n\n") if b.strip()]
    sections = []
    
    current_title = title
    buffer = []

    for block in raw_blocks:
        if len(block) < 80 and not block.endswith((".", ":", ";")):
            # Likely a sub-header
            if buffer:
                sections.append(
                    ParsedSection(
                        text="\n\n".join(buffer),
                        section_title=current_title,
                        metadata={"url": url}
                    )
                )
                buffer = []
            current_title = block
        else:
            buffer.append(block)

    if buffer:
        sections.append(
            ParsedSection(
                text="\n\n".join(buffer),
                section_title=current_title,
                metadata={"url": url}
            )
        )

    if not sections and cleaned_text:
        sections.append(
            ParsedSection(
                text=cleaned_text,
                section_title=title,
                metadata={"url": url}
            )
        )

    return ParsedDocument(
        doc_id=doc_id,
        filename=title,
        source_type="web",
        source_url=url,
        sections=sections,
        total_pages=None,
        raw_text=cleaned_text,
        metadata={
            "url": url,
            "title": title,
            "author": metadata.author if metadata else None,
            "date": metadata.date if metadata else None,
            "section_count": len(sections),
        }
    )
