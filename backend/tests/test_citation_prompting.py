import re
import sys
from pathlib import Path

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add backend to sys.path
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

from app.schemas.query import RetrievedChunk
from app.services.synthesizer import SynthesizerService, CITATION_SYSTEM_PROMPT

def test_source_context_formatting():
    service = SynthesizerService()

    chunks = [
        RetrievedChunk(
            chunk_id="chunk_pdf_01",
            doc_id="doc_pdf",
            filename="quarterly_report.pdf",
            source_type="pdf",
            page_number=14,
            section_title="Operating Results",
            text="Operating margin expanded by 14% in Q3 driven by enterprise cloud contracts.",
            score=0.92,
        ),
        RetrievedChunk(
            chunk_id="chunk_web_02",
            doc_id="doc_web",
            filename="Product Release",
            source_type="web",
            source_url="https://docunexus.ai/news/v1",
            section_title="Release Notes",
            text="DocuNexus v1.0 was officially launched across all European enterprise clusters.",
            score=0.88,
        ),
        RetrievedChunk(
            chunk_id="chunk_docx_03",
            doc_id="doc_docx",
            filename="handbook.docx",
            source_type="docx",
            section_title="Leave Policies",
            text="Paid annual leave is credited at 2.5 days per month of service.",
            score=0.75,
        ),
    ]

    context_str, source_map = service.format_sources_context(chunks)

    # 1. Verify source map indexing
    assert len(source_map) == 3
    assert source_map[1].chunk_id == "chunk_pdf_01"
    assert source_map[2].chunk_id == "chunk_web_02"
    assert source_map[3].chunk_id == "chunk_docx_03"
    print("[PASS] Source mapping verified (1 -> PDF, 2 -> Web, 3 -> DOCX)!")

    # 2. Verify provenance tags in formatted context
    assert "[Source 1] (File: quarterly_report.pdf, Page: 14, Section: Operating Results):" in context_str
    assert "[Source 2] (File: Product Release, Section: Release Notes, URL: https://docunexus.ai/news/v1):" in context_str
    assert "[Source 3] (File: handbook.docx, Section: Leave Policies):" in context_str
    print("[PASS] Multi-format provenance headers in prompt context verified!")

def test_citation_prompt_construction():
    service = SynthesizerService()
    query = "What caused operating margin expansion in Q3?"
    context = "[Source 1] (File: report.pdf): Operating margin expanded by 14%."

    prompt = service.build_citation_prompt(query, context)
    assert query in prompt
    assert context in prompt
    assert "cite every fact using [1], [2]" in prompt
    assert len(CITATION_SYSTEM_PROMPT) > 100
    print("[PASS] Citation-constrained prompt template verified!")

def test_citation_answer_generation():
    service = SynthesizerService()

    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            doc_id="d1",
            filename="financials.pdf",
            source_type="pdf",
            page_number=4,
            text="Operating margin expanded by 14% in the third quarter.",
            score=0.9,
        ),
        RetrievedChunk(
            chunk_id="c2",
            doc_id="d2",
            filename="expansion.docx",
            source_type="docx",
            section_title="EMEA",
            text="EMEA region reached profitability ahead of schedule.",
            score=0.85,
        ),
    ]

    query = "How did operating margin and EMEA perform?"
    answer, source_map = service.generate_answer(query, chunks)

    # Verify inline citations exist
    citations_found = re.findall(r'\[(\d+)\]', answer)
    assert len(citations_found) > 0, f"Expected citations in answer, got: '{answer}'"

    # Verify all citation numbers match valid source indices
    valid_indices = set(str(k) for k in source_map.keys())
    for cite_num in citations_found:
        assert cite_num in valid_indices, f"Invalid citation [{cite_num}] not in {valid_indices}"

    print(f"[PASS] Answer generated with verifiable inline citations: '{answer}' (Citations: {citations_found})!")

if __name__ == "__main__":
    print("Testing Citation-Constrained Prompting (Step 5.1)...")
    test_source_context_formatting()
    test_citation_prompt_construction()
    test_citation_answer_generation()
    print("ALL Citation Prompting tests completed successfully!")
