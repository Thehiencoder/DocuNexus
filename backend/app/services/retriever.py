import logging
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from llama_index.core.schema import TextNode, NodeWithScore
from app.core.config import settings
from app.core.qdrant_client import get_qdrant_client
from app.services.embeddings import embedding_service
from app.schemas.query import RetrievedChunk

logger = logging.getLogger(__name__)

class HybridRetrieverService:
    """
    Executes Hybrid Search combining Dense Vector (Google Gemini 768d)
    and Sparse Vector (FastEmbed BM25) with Reciprocal Rank Fusion (RRF) in Qdrant.
    """

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        collection_name: Optional[str] = None,
        default_top_k: int = 25,
    ):
        self._client = client
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.default_top_k = default_top_k

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = get_qdrant_client()
        return self._client

    def set_client(self, client: QdrantClient):
        """Allows injecting an in-memory client or mock for testing."""
        self._client = client

    def _build_filter(self, doc_ids: Optional[List[str]] = None) -> Optional[rest_models.Filter]:
        """Builds Qdrant payload filter for scoping search to specific documents."""
        if not doc_ids:
            return None
        if len(doc_ids) == 1:
            return rest_models.Filter(
                must=[
                    rest_models.FieldCondition(
                        key="doc_id",
                        match=rest_models.MatchValue(value=doc_ids[0]),
                    )
                ]
            )
        return rest_models.Filter(
            must=[
                rest_models.FieldCondition(
                    key="doc_id",
                    match=rest_models.MatchAny(any=doc_ids),
                )
            ]
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Executes hybrid search (BM25 + Dense) via Reciprocal Rank Fusion (RRF).
        Returns top candidate chunks formatted as RetrievedChunk objects.
        """
        limit = top_k or self.default_top_k
        query_text = query.strip()
        if not query_text:
            return []

        # 1. Embed query into dual representations
        dual_emb = embedding_service.embed_text(query_text, is_query=True)
        qdrant_filter = self._build_filter(doc_ids)

        # 2. Build prefetch queries for Dense and Sparse spaces
        prefetch_queries = [
            rest_models.Prefetch(
                query=dual_emb.dense,
                using="dense",
                filter=qdrant_filter,
                limit=limit,
            )
        ]

        if dual_emb.sparse_indices and len(dual_emb.sparse_indices) > 0:
            prefetch_queries.append(
                rest_models.Prefetch(
                    query=rest_models.SparseVector(
                        indices=dual_emb.sparse_indices,
                        values=dual_emb.sparse_values,
                    ),
                    using="sparse",
                    filter=qdrant_filter,
                    limit=limit,
                )
            )

        # 3. Query Qdrant with Reciprocal Rank Fusion (RRF)
        try:
            if len(prefetch_queries) > 1:
                # True hybrid search with RRF
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=prefetch_queries,
                    query=rest_models.FusionQuery(fusion=rest_models.Fusion.RRF),
                    limit=limit,
                    with_payload=True,
                )
            else:
                # Fallback to dense query if no sparse keywords
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=dual_emb.dense,
                    using="dense",
                    query_filter=qdrant_filter,
                    limit=limit,
                    with_payload=True,
                )
        except Exception as e:
            logger.error(f"Hybrid retrieval query error on collection '{self.collection_name}': {e}")
            return []

        # 4. Map Qdrant points into structured RetrievedChunks
        results: List[RetrievedChunk] = []
        for point in response.points:
            payload = point.payload or {}
            chunk = RetrievedChunk(
                chunk_id=payload.get("chunk_id", str(point.id)),
                doc_id=payload.get("doc_id", ""),
                filename=payload.get("filename", ""),
                source_type=payload.get("source_type", ""),
                source_url=payload.get("source_url"),
                page_number=payload.get("page_number"),
                section_title=payload.get("section_title"),
                text=payload.get("text", ""),
                score=float(point.score) if point.score is not None else 0.0,
                char_start=payload.get("char_start", 0),
                char_end=payload.get("char_end", 0),
                chunk_index=payload.get("chunk_index", 0),
                metadata=payload,
            )
            results.append(chunk)

        return results

    def retrieve_nodes(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_ids: Optional[List[str]] = None,
    ) -> List[NodeWithScore]:
        """
        Retrieves candidates and transforms them into LlamaIndex NodeWithScore objects,
        ready for LlamaIndex postprocessors (Cohere Rerank) and synthesizers.
        """
        chunks = self.search(query=query, top_k=top_k, doc_ids=doc_ids)
        nodes_with_score: List[NodeWithScore] = []

        for chunk in chunks:
            node = TextNode(
                id_=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                excluded_embed_metadata_keys=[
                    "chunk_id",
                    "doc_id",
                    "char_start",
                    "char_end",
                    "chunk_index",
                ],
                excluded_llm_metadata_keys=[
                    "chunk_id",
                    "doc_id",
                    "char_start",
                    "char_end",
                    "chunk_index",
                ],
            )
            nodes_with_score.append(NodeWithScore(node=node, score=chunk.score))

        return nodes_with_score

hybrid_retriever = HybridRetrieverService()
