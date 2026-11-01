import sys
from pathlib import Path

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add backend to sys.path
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

from qdrant_client import QdrantClient
from llama_index.core.schema import TextNode, NodeWithScore
from app.schemas.query import RetrievedChunk
from app.services.reranker import RerankerService, reranker_service
from app.services.vector_store import VectorStoreService, vector_store
from app.services.retriever import HybridRetrieverService, hybrid_retriever
from starlette.testclient import TestClient
from app.main import app

def test_reranker_service_pruning_and_scoring():
    service = RerankerService(top_n=3)

    query = "What were the primary drivers of operating margin expansion in Q3?"

    # 5 test chunks with varying relevance
    chunks = [
        RetrievedChunk(
            chunk_id="c_unrelated_01",
            doc_id="doc_hr",
            filename="handbook.txt",
            source_type="txt",
            text="Employee benefits include dental, vision, and paid parental leave.",
            score=0.45,
        ),
        RetrievedChunk(
            chunk_id="c_partial_02",
            doc_id="doc_ops",
            filename="facility_ops.txt",
            source_type="txt",
            text="Office facilities saw increased electricity costs during the summer season.",
            score=0.52,
        ),
        RetrievedChunk(
            chunk_id="c_target_03",
            doc_id="doc_fin",
            filename="quarterly_report.pdf",
            source_type="pdf",
            page_number=4,
            text="Operating margin expansion in Q3 was driven primarily by cloud infrastructure automation and supply chain optimizations.",
            score=0.68,
        ),
        RetrievedChunk(
            chunk_id="c_semi_04",
            doc_id="doc_fin",
            filename="quarterly_report.pdf",
            source_type="pdf",
            page_number=2,
            text="Total revenue for the quarter was in line with guidance.",
            score=0.60,
        ),
        RetrievedChunk(
            chunk_id="c_vague_05",
            doc_id="doc_memo",
            filename="memo.docx",
            source_type="docx",
            text="General market trends in Europe remain steady.",
            score=0.35,
        ),
    ]

    # Rerank 5 chunks down to top 3
    reranked = service.rerank_chunks(query=query, chunks=chunks, top_n=3)

    assert len(reranked) == 3, f"Expected 3 reranked chunks, got {len(reranked)}"
    # Target chunk with exact answer should be rank #1
    top_chunk = reranked[0]
    assert top_chunk.chunk_id == "c_target_03"
    assert "Operating margin expansion in Q3 was driven primarily" in top_chunk.text
    assert top_chunk.score > reranked[1].score
    print(f"[PASS] Reranker pruned 5 candidates down to top {len(reranked)}; target chunk ranked #1 with score {top_chunk.score:.4f}!")

def test_reranker_nodes_transformation():
    service = RerankerService(top_n=2)
    query = "GPU cluster memory requirements"

    nodes = [
        NodeWithScore(
            node=TextNode(id_="n1", text="Each GPU node is equipped with 80GB of HBM3 high-bandwidth memory for LLM training."),
            score=0.4,
        ),
        NodeWithScore(
            node=TextNode(id_="n2", text="Company cafeteria hours are 8am to 4pm Monday through Friday."),
            score=0.3,
        ),
        NodeWithScore(
            node=TextNode(id_="n3", text="Network cabling standards require CAT6e connections."),
            score=0.35,
        ),
    ]

    reranked_nodes = service.rerank_nodes(query=query, nodes=nodes, top_n=2)
    assert len(reranked_nodes) == 2
    assert reranked_nodes[0].node.node_id == "n1"
    print("[PASS] LlamaIndex NodeWithScore reranking verified!")

def test_api_rerank_endpoint():
    test_client = TestClient(app)

    # Inject in-memory client
    mem_client = QdrantClient(":memory:")
    vector_store.set_client(mem_client)
    hybrid_retriever.set_client(mem_client)
    vector_store.ensure_collection()

    # Ingest sample markdown document
    content = b"""# Cloud Infrastructure Q3
Operating margin expansion in Q3 achieved a record 28.5% due to data center server efficiency.

# Facilities
Water usage remained consistent with previous quarters.
"""
    test_client.post("/api/v1/documents/upload", files={"file": ("infra_q3.md", content, "text/markdown")})

    # Call /api/v1/query/rerank
    res = test_client.post(
        "/api/v1/query/rerank",
        json={
            "query": "operating margin expansion efficiency",
            "top_k": 10,
            "top_n": 2,
        },
    )
    assert res.status_code == 200, f"Rerank failed: {res.text}"
    data = res.json()
    assert data["initial_candidates_count"] >= 1
    assert data["reranked_count"] <= 2
    assert len(data["results"]) > 0
    assert "operating margin expansion" in data["results"][0]["text"].lower()
    print(f"[PASS] API /api/v1/query/rerank verified! ({data['initial_candidates_count']} candidates -> {data['reranked_count']} reranked)")

if __name__ == "__main__":
    print("Testing Cohere Reranker Integration (Step 4.2)...")
    test_reranker_service_pruning_and_scoring()
    test_reranker_nodes_transformation()
    test_api_rerank_endpoint()
    print("ALL Cohere Reranker tests completed successfully!")
