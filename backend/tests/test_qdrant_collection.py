import sys
from pathlib import Path

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add backend to sys.path
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from app.core.qdrant_client import init_qdrant_collection, get_collection_status

def test_qdrant_named_vectors_initialization():
    client = QdrantClient(":memory:")
    col_name = "test_knowledge_base"
    
    success = init_qdrant_collection(client=client, collection_name=col_name, dense_dim=768)
    assert success is True, "Collection initialization failed"

    col_info = client.get_collection(collection_name=col_name)
    assert col_info is not None

    # Verify dense named vector config
    vectors_cfg = col_info.config.params.vectors
    assert "dense" in vectors_cfg
    assert vectors_cfg["dense"].size == 768
    assert vectors_cfg["dense"].distance == rest_models.Distance.COSINE
    print("[PASS] Named Vector 'dense' verified (768-dim, Cosine distance)!")

    # Verify sparse named vector config (BM25)
    sparse_cfg = col_info.config.params.sparse_vectors
    assert "sparse" in sparse_cfg
    assert sparse_cfg["sparse"].modifier == rest_models.Modifier.IDF
    print("[PASS] Named Vector 'sparse' verified (BM25 with IDF modifier)!")

    # Test dual vector upsert into named vector spaces
    client.upsert(
        collection_name=col_name,
        points=[
            rest_models.PointStruct(
                id=1,
                vector={
                    "dense": [0.05] * 768,
                    "sparse": rest_models.SparseVector(indices=[10, 42, 88], values=[1.2, 0.9, 2.5]),
                },
                payload={
                    "doc_id": "doc_test_01",
                    "source_type": "pdf",
                    "page_number": 1,
                    "text": "Quarterly operating margins expanded by 14%.",
                },
            )
        ],
    )
    col_info = client.get_collection(collection_name=col_name)
    assert col_info.points_count == 1
    print("[PASS] Dual vector upsert verified successfully (points_count: 1)!")

    # Test query on dense named vector
    dense_results = client.query_points(
        collection_name=col_name,
        using="dense",
        query=[0.05] * 768,
        limit=1,
    )
    assert len(dense_results.points) == 1
    assert dense_results.points[0].payload["doc_id"] == "doc_test_01"
    print("[PASS] Query on Named Vector 'dense' returned matching point!")

def test_live_qdrant_connection():
    status = get_collection_status()
    if status["status"] == "connected":
        print(f"[PASS] Connected to Live Qdrant instance on localhost:6333 ({status['points_count']} points)!")
    else:
        print(f"[NOTE] Live Qdrant Docker is {status['status']} (Docker container runs via 'docker compose up -d').")

if __name__ == "__main__":
    print("Testing Qdrant Collection Initialization & Named Vectors...")
    test_qdrant_named_vectors_initialization()
    test_live_qdrant_connection()
    print("ALL Qdrant collection tests completed successfully!")
