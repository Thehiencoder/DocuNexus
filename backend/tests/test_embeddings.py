import sys
from pathlib import Path

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add backend to sys.path
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

from app.services.embeddings import embedding_service, DualEmbedding

def test_single_text_dual_embedding():
    text = "DocuNexus provides multi-document hybrid search with BM25 and vector similarity."
    dual = embedding_service.embed_text(text, is_query=False)

    # 1. Verify Dense Vector
    assert isinstance(dual.dense, list)
    assert len(dual.dense) == 768, f"Expected 768 dimensions, got {len(dual.dense)}"
    assert all(isinstance(x, (float, int)) for x in dual.dense)
    print(f"[PASS] Dense embedding generated successfully (dim: {len(dual.dense)})!")

    # 2. Verify Sparse BM25 Vector
    assert isinstance(dual.sparse_indices, list)
    assert isinstance(dual.sparse_values, list)
    assert len(dual.sparse_indices) == len(dual.sparse_values)
    assert len(dual.sparse_indices) > 0, "Sparse BM25 vector should not be empty"
    print(f"[PASS] Sparse BM25 embedding generated successfully ({len(dual.sparse_indices)} active tokens)!")

    # 3. Verify Qdrant Vector Conversion
    qdrant_payload = dual.to_qdrant_vectors()
    assert "dense" in qdrant_payload
    assert "sparse" in qdrant_payload
    assert len(qdrant_payload["dense"]) == 768
    print("[PASS] Conversion to Qdrant named vector format verified!")

def test_batch_embedding():
    texts = [
        "First chapter introduces machine learning fundamentals.",
        "Second chapter dives into transformer architectures and self-attention.",
        "Third chapter covers vector databases and BM25 hybrid indexing.",
    ]
    batch_results = embedding_service.embed_batch(texts)
    assert len(batch_results) == 3
    for idx, res in enumerate(batch_results):
        assert len(res.dense) == 768
        assert len(res.sparse_indices) > 0
    print("[PASS] Batch embedding verified across multiple passages!")

if __name__ == "__main__":
    print("Testing Dual Embedding Generator (Dense + Sparse BM25)...")
    test_single_text_dual_embedding()
    test_batch_embedding()
    print("ALL embedding tests completed successfully!")
