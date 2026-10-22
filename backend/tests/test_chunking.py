import sys
from pathlib import Path

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add backend to sys.path
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

from app.schemas.document import ParsedDocument, ParsedSection
from app.services.chunking import ChunkingService

def test_pdf_page_metadata_preservation():
    # Simulate a 3-page parsed PDF
    doc = ParsedDocument(
        doc_id="doc_multi_page_123",
        filename="quarterly_report.pdf",
        source_type="pdf",
        sections=[
            ParsedSection(
                text="Financial Highlights for Q1. Revenue increased by 18% driven by cloud growth.",
                page_number=1,
                section_title="Page 1",
            ),
            ParsedSection(
                text="Operating Margins and Expansion. Margins improved by 350 basis points across EMEA.",
                page_number=2,
                section_title="Page 2",
            ),
            ParsedSection(
                text="Future Outlook and Guidance. Full-year revenue growth is projected at 20-22%.",
                page_number=3,
                section_title="Page 3",
            ),
        ],
        total_pages=3,
        raw_text="Full text...",
    )

    chunker = ChunkingService(chunk_size=100, chunk_overlap=20)
    nodes, chunks = chunker.chunk_document(doc)

    assert len(nodes) == len(chunks)
    assert len(nodes) >= 3

    # Check page numbers are strictly preserved
    page_1_nodes = [n for n in nodes if n.metadata["page_number"] == 1]
    page_2_nodes = [n for n in nodes if n.metadata["page_number"] == 2]
    page_3_nodes = [n for n in nodes if n.metadata["page_number"] == 3]

    assert len(page_1_nodes) >= 1
    assert len(page_2_nodes) >= 1
    assert len(page_3_nodes) >= 1

    # Verify character offsets and attributes
    for node in nodes:
        assert node.metadata["doc_id"] == "doc_multi_page_123"
        assert node.metadata["filename"] == "quarterly_report.pdf"
        assert node.metadata["source_type"] == "pdf"
        assert node.metadata["char_end"] > node.metadata["char_start"]
        assert len(node.text) > 0

    print("[PASS] Multi-page PDF chunking and page metadata preservation verified!")

def test_docx_section_title_preservation():
    doc = ParsedDocument(
        doc_id="doc_docx_456",
        filename="product_spec.docx",
        source_type="docx",
        sections=[
            ParsedSection(
                text="System Architecture Overview and Core Components.",
                section_title="1. Introduction",
            ),
            ParsedSection(
                text="Database Schema and Named Vectors specification for Qdrant.",
                section_title="2. Data Layer",
            ),
        ],
        raw_text="Full text...",
    )

    chunker = ChunkingService(chunk_size=120, chunk_overlap=20)
    nodes, chunks = chunker.chunk_document(doc)

    assert any(n.metadata["section_title"] == "1. Introduction" for n in nodes)
    assert any(n.metadata["section_title"] == "2. Data Layer" for n in nodes)
    print("[PASS] DOCX section title metadata preservation verified!")

if __name__ == "__main__":
    print("Running chunking and metadata tests...")
    test_pdf_page_metadata_preservation()
    test_docx_section_title_preservation()
    print("ALL chunking tests completed successfully!")
