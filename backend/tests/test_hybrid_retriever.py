import sys
from pathlib import Path

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add backend to sys.path
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

from qdrant_client import QdrantClient
from app.schemas.document import ParsedDocument, ParsedSection
from app.services.chunking import ChunkingService
from app.services.vector_store import VectorStoreService, vector_store
from app.services.retriever import HybridRetrieverService, hybrid_retriever
from starlette.testclient import TestClient
from app.main import app

def test_hybrid_search_lexical_and_semantic():
    mem_client = QdrantClient(":memory:")
    col_name = "test_hybrid_eval_kb"

    vs = VectorStoreService(client=mem_client, collection_name=col_name)
    retriever = HybridRetrieverService(client=mem_client, collection_name=col_name, default_top_k=5)

    # Ingest 3 distinct documents
    docs = [
        ParsedDocument(
            doc_id="doc_tech_01",
            filename="system_errors.log",
            source_type="txt",
            sections=[
                ParsedSection(
                    text="Critical failure code ERR_AUTH_GATEWAY_TIMEOUT detected on port 9443.",
                    section_title="Error Log",
                )
            ],
            raw_text="...",
        ),
        ParsedDocument(
            doc_id="doc_finance_02",
            filename="q3_earnings.pdf",
            source_type="pdf",
            sections=[
                ParsedSection(
                    text="Consolidated financial results showed significant revenue growth and margin acceleration.",
                    page_number=3,
                    section_title="Operating Results",
                )
            ],
            raw_text="...",
        ),
        ParsedDocument(
            doc_id="doc_ai_03",
            filename="ml_whitepaper.md",
            source_type="txt",
            sections=[
                ParsedSection(
                    text="Deep learning architectures rely heavily on GPU accelerators for matrix multiplication.",
                    section_title="Hardware Requirements",
                )
            ],
            raw_text="...",
        ),
    ]

    chunker = ChunkingService()
    all_chunks = []
    for d in docs:
        _, chs = chunker.chunk_document(d)
        all_chunks.extend(chs)

    vs.upsert_chunks(all_chunks)
    assert vs.count_document_chunks() == 3
    print("[PASS] Ingested 3 test documents into in-memory Qdrant.")

    # Test 1: Exact keyword match (BM25 lexical strength)
    keyword_query = "ERR_AUTH_GATEWAY_TIMEOUT"
    keyword_results = retriever.search(query=keyword_query, top_k=3)
    assert len(keyword_results) > 0
    top_hit = keyword_results[0]
    assert top_hit.doc_id == "doc_tech_01"
    assert "ERR_AUTH_GATEWAY_TIMEOUT" in top_hit.text
    print(f"[PASS] Exact keyword query ('{keyword_query}') surfaced target document #1 via BM25 (score: {top_hit.score:.4f})!")

    # Test 2: Natural language query (Hybrid RRF fusion)
    finance_query = "financial revenue growth and margin results"
    finance_results = retriever.search(query=finance_query, top_k=3)
    assert len(finance_results) > 0
    top_fin_hit = finance_results[0]
    assert top_fin_hit.doc_id == "doc_finance_02"
    assert top_fin_hit.page_number == 3
    print(f"[PASS] Financial query ('{finance_query}') surfaced target document #1 via Hybrid RRF (score: {top_fin_hit.score:.4f})!")

    # Test 3: LlamaIndex NodeWithScore transformation
    nodes_with_score = retriever.retrieve_nodes(query="GPU accelerators for deep learning", top_k=2)
    assert len(nodes_with_score) > 0
    assert hasattr(nodes_with_score[0], "node")
    assert hasattr(nodes_with_score[0], "score")
    assert nodes_with_score[0].node.metadata["doc_id"] == "doc_ai_03"
    print("[PASS] Transformation to LlamaIndex NodeWithScore verified!")

    # Test 4: Document ID filtering
    filtered_results = retriever.search(query="matrix multiplication", top_k=3, doc_ids=["doc_finance_02"])
    assert all(r.doc_id == "doc_finance_02" for r in filtered_results)
    assert not any(r.doc_id == "doc_ai_03" for r in filtered_results)
    print("[PASS] Scoped document filtering verified (doc_ai_03 excluded, doc_finance_02 scoped)!")

def test_api_hybrid_search_endpoint():
    test_client = TestClient(app)

    # Inject in-memory client into global services for test
    mem_client = QdrantClient(":memory:")
    vector_store.set_client(mem_client)
    hybrid_retriever.set_client(mem_client)
    vector_store.ensure_collection()

    # Upload document through API
    content = b"# Q3 Financial Performance\nOperating profit expanded by 24% year-over-year."
    test_client.post("/api/v1/documents/upload", files={"file": ("finance.md", content, "text/markdown")})

    # Search via /api/v1/query/search
    res = test_client.post(
        "/api/v1/query/search",
        json={"query": "operating profit expansion", "top_k": 5},
    )
    assert res.status_code == 200, f"Search failed: {res.text}"
    data = res.json()
    assert data["total_results"] >= 1
    assert "operating profit" in data["results"][0]["text"].lower()
    print(f"[PASS] API /api/v1/query/search returned {data['total_results']} hybrid candidates!")

if __name__ == "__main__":
    print("Testing Hybrid Search (Dense + BM25 with Reciprocal Rank Fusion)...")
    test_hybrid_search_lexical_and_semantic()
    test_api_hybrid_search_endpoint()
    print("ALL Hybrid Retrieval tests completed successfully!")
