import io
import sys
from pathlib import Path

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add backend to sys.path
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

from qdrant_client import QdrantClient
from app.schemas.document import DocumentChunk, ParsedDocument, ParsedSection
from app.services.chunking import ChunkingService
from app.services.vector_store import VectorStoreService, vector_store
from starlette.testclient import TestClient
from app.main import app

def test_vector_store_upsert_and_delete_in_memory():
    """Tests Qdrant vector store point insertion, payload inspection, and purging in memory."""
    mem_client = QdrantClient(":memory:")
    col_name = "test_lifecycle_kb"
    service = VectorStoreService(client=mem_client, collection_name=col_name)

    # 1. Create sample document chunks
    doc = ParsedDocument(
        doc_id="doc_lifecycle_001",
        filename="financial_overview.pdf",
        source_type="pdf",
        sections=[
            ParsedSection(
                text="In Q3, consolidated operating revenue rose by 16.4% year-over-year.",
                page_number=1,
                section_title="Executive Summary",
            ),
            ParsedSection(
                text="Gross margins expanded 210 bps due to efficiencies in cloud infrastructure.",
                page_number=2,
                section_title="Margin Analysis",
            ),
        ],
        total_pages=2,
        raw_text="Full text...",
    )

    chunker = ChunkingService(chunk_size=120, chunk_overlap=20)
    nodes, chunks = chunker.chunk_document(doc)
    assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"

    # 2. Upsert into Qdrant
    upserted_count = service.upsert_chunks(chunks)
    assert upserted_count == 2
    print(f"[PASS] Successfully indexed {upserted_count} chunks into Qdrant collection!")

    # 3. Verify count
    total_in_db = service.count_document_chunks("doc_lifecycle_001")
    assert total_in_db == 2
    print(f"[PASS] Point count verified via doc_id filter: {total_in_db} points.")

    # 4. Inspect stored points (payload & vectors)
    scroll_res = mem_client.scroll(
        collection_name=col_name,
        limit=2,
        with_payload=True,
        with_vectors=True,
    )
    points = scroll_res[0]
    assert len(points) == 2

    for pt in points:
        payload = pt.payload
        assert payload["doc_id"] == "doc_lifecycle_001"
        assert payload["filename"] == "financial_overview.pdf"
        assert payload["source_type"] == "pdf"
        assert payload["page_number"] in [1, 2]
        assert "text" in payload

        # Check vectors
        vectors = pt.vector
        assert "dense" in vectors
        assert len(vectors["dense"]) == 768
        assert "sparse" in vectors
        assert hasattr(vectors["sparse"], "indices")
        assert len(vectors["sparse"].indices) > 0

    print("[PASS] Point payloads and dual named vectors (dense 768d + sparse BM25) verified!")

    # 5. Delete document points
    del_ok = service.delete_document_chunks("doc_lifecycle_001")
    assert del_ok is True
    remaining_count = service.count_document_chunks("doc_lifecycle_001")
    assert remaining_count == 0
    print("[PASS] Document chunks successfully purged from Qdrant by doc_id filter!")

def test_api_e2e_with_vector_indexing():
    """Tests the full API flow from upload -> vector indexing -> list -> delete."""
    test_client = TestClient(app)

    # Inject in-memory client into global vector_store singleton for clean test
    mem_client = QdrantClient(":memory:")
    vector_store.set_client(mem_client)
    vector_store.ensure_collection()

    content = b"# Architecture Highlights\nDocuNexus uses hybrid search combining dense vectors and BM25."
    files = {"file": ("test_doc_arch.md", content, "text/markdown")}
    res = test_client.post("/api/v1/documents/upload", files=files)
    assert res.status_code == 201, f"Failed: {res.text}"
    data = res.json()
    doc_id = data["doc_id"]
    print(f"[PASS] API uploaded and indexed document: {doc_id}")

    # Verify points in Qdrant
    point_count = vector_store.count_document_chunks(doc_id)
    assert point_count >= 1
    print(f"[PASS] Vector store has {point_count} points indexed for {doc_id}")

    # List documents
    list_res = test_client.get("/api/v1/documents")
    assert list_res.status_code == 200
    docs = list_res.json()
    assert any(d["doc_id"] == doc_id for d in docs)
    print(f"[PASS] GET /api/v1/documents returned {len(docs)} documents.")

    # Delete document
    del_res = test_client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200
    print(f"[PASS] DELETE /api/v1/documents/{doc_id} returned 200.")

    # Verify points were purged from Qdrant
    assert vector_store.count_document_chunks(doc_id) == 0
    print("[PASS] Verified points purged from vector store after API delete.")

if __name__ == "__main__":
    print("Testing Vector Store Ingestion & Document Lifecycle...")
    test_vector_store_upsert_and_delete_in_memory()
    test_api_e2e_with_vector_indexing()
    print("ALL Vector Store Lifecycle tests completed successfully!")
