import logging
import hashlib
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from app.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class DualEmbedding:
    """Represents the paired dense and sparse vectors for a chunk or query."""
    dense: List[float]
    sparse_indices: List[int]
    sparse_values: List[float]

    def to_qdrant_vectors(self) -> Dict[str, Any]:
        """Formats the dual embedding for Qdrant Named Vectors."""
        from qdrant_client.http import models as rest_models
        return {
            "dense": self.dense,
            "sparse": rest_models.SparseVector(
                indices=self.sparse_indices,
                values=self.sparse_values,
            ),
        }

class EmbeddingService:
    """
    Produces dual representations for Hybrid Search:
      1. Dense Vector: Google Gemini text-embedding-004 (768-dim semantic similarity).
      2. Sparse Vector: FastEmbed Qdrant/bm25 (lexical exact keyword scoring).
    """

    def __init__(self):
        self._sparse_model = None
        self._gemini_initialized = False
        self._init_gemini()

    def _init_gemini(self):
        if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your_gemini_api_key_here":
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self._gemini_initialized = True
                logger.info("Google Gemini Embeddings initialized with API key.")
            except Exception as e:
                logger.warning(f"Failed to configure Google Gemini: {e}")
        else:
            logger.info("GEMINI_API_KEY not configured. Local fallback dense generator will be used.")

    @property
    def sparse_model(self):
        """Lazy load FastEmbed BM25 model on first use."""
        if self._sparse_model is None:
            from fastembed import SparseTextEmbedding
            logger.info("Loading FastEmbed BM25 sparse embedding model (Qdrant/bm25)...")
            self._sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        return self._sparse_model

    def _generate_dense_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """
        Generates dense vector using Google Gemini text-embedding-004.
        Falls back to deterministic local dense embedding if offline/unconfigured.
        """
        if self._gemini_initialized:
            import google.generativeai as genai
            task_type = "retrieval_query" if is_query else "retrieval_document"
            try:
                res = genai.embed_content(
                    model=settings.EMBEDDING_MODEL,
                    content=text,
                    task_type=task_type,
                )
                if "embedding" in res:
                    return res["embedding"]
            except Exception as e:
                logger.warning(f"Gemini API call failed, using fallback: {e}")

        # Deterministic 768-dim local fallback vector for offline testing
        return self._generate_fallback_dense(text, dim=settings.EMBEDDING_DIM)

    def _generate_fallback_dense(self, text: str, dim: int = 768) -> List[float]:
        """Generates a normalized deterministic float vector from text hash."""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        raw = []
        for i in range(dim):
            byte_val = h[i % len(h)]
            raw.append(((byte_val / 255.0) * 2.0) - 1.0)
        norm = sum(x * x for x in raw) ** 0.5 or 1.0
        return [round(x / norm, 6) for x in raw]

    def _generate_sparse_embedding(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Generates BM25 sparse token indices and weights via FastEmbed.
        """
        try:
            embeddings = list(self.sparse_model.embed([text]))
            if embeddings:
                emb = embeddings[0]
                return emb.indices.tolist(), [round(float(v), 5) for v in emb.values]
        except Exception as e:
            logger.warning(f"FastEmbed BM25 generation failed, using keyword fallback: {e}")

        # Lightweight fallback token frequency if FastEmbed ONNX encounters issues
        words = set(text.lower().split())
        indices = [abs(hash(w)) % 100000 for w in words]
        values = [1.0] * len(indices)
        return indices, values

    def embed_text(self, text: str, is_query: bool = False) -> DualEmbedding:
        """Embeds a single passage or search query into dual vectors."""
        dense = self._generate_dense_embedding(text, is_query=is_query)
        sparse_indices, sparse_values = self._generate_sparse_embedding(text)
        return DualEmbedding(
            dense=dense,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
        )

    def embed_batch(self, texts: List[str], is_query: bool = False) -> List[DualEmbedding]:
        """Embeds a batch of passages."""
        return [self.embed_text(t, is_query=is_query) for t in texts]

# Singleton instance
embedding_service = EmbeddingService()
