import logging
import uuid
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from app.core.config import settings
from app.core.qdrant_client import get_qdrant_client, init_qdrant_collection
from app.services.embeddings import embedding_service
from app.schemas.document import DocumentChunk

logger = logging.getLogger(__name__)

class VectorStoreService:
    """
    Manages vector ingestion, payload indexing, point lifecycle,
    and deletion within Qdrant collection using Named Vectors.
    """

    def __init__(self, client: Optional[QdrantClient] = None, collection_name: Optional[str] = None):
        self._client = client
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = get_qdrant_client()
        return self._client

    def set_client(self, client: QdrantClient):
        """Allows injecting an in-memory client or mock for testing."""
        self._client = client

    def ensure_collection(self) -> bool:
        """Ensures the collection exists with dual named vectors."""
        return init_qdrant_collection(client=self.client, collection_name=self.collection_name)

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
    ) -> int:
        """
        Embeds a list of DocumentChunks using dual embeddings (dense 768d + sparse BM25)
        and upserts them into Qdrant as points with full metadata payload.
        Returns the number of points upserted.
        """
        if not chunks:
            return 0

        self.ensure_collection()

        texts = [chunk.text for chunk in chunks]
        embeddings = embedding_service.embed_batch(texts, is_query=False)

        points: List[rest_models.PointStruct] = []
        for chunk, dual_emb in zip(chunks, embeddings):
            # Deterministic UUID derived from chunk_id ensures idempotency
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))

            payload: Dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "filename": chunk.filename,
                "source_type": chunk.source_type,
                "source_url": chunk.source_url,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "chunk_index": chunk.chunk_index,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "text": chunk.text,
            }
            # Merge custom metadata if any
            if chunk.metadata:
                for k, v in chunk.metadata.items():
                    if k not in payload:
                        payload[k] = v

            vector_dict = dual_emb.to_qdrant_vectors()

            point = rest_models.PointStruct(
                id=point_id,
                vector=vector_dict,
                payload=payload,
            )
            points.append(point)

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            logger.info(f"Successfully upserted {len(points)} chunks for doc '{chunks[0].doc_id}' into Qdrant.")
            return len(points)
        except Exception as e:
            logger.error(f"Failed to upsert chunks into Qdrant: {e}")
            raise e

    def delete_document_chunks(self, doc_id: str) -> bool:
        """
        Purges all chunk points in Qdrant matching the doc_id payload filter.
        """
        try:
            filter_selector = rest_models.FilterSelector(
                filter=rest_models.Filter(
                    must=[
                        rest_models.FieldCondition(
                            key="doc_id",
                            match=rest_models.MatchValue(value=doc_id),
                        )
                    ]
                )
            )
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_selector,
                wait=True,
            )
            logger.info(f"Purged all points for doc_id '{doc_id}' from Qdrant.")
            return True
        except Exception as e:
            logger.warning(f"Could not purge points for doc_id '{doc_id}' from Qdrant: {e}")
            return False

    def count_document_chunks(self, doc_id: Optional[str] = None) -> int:
        """
        Counts the number of points in Qdrant, optionally filtered by doc_id.
        """
        try:
            filter_obj = None
            if doc_id:
                filter_obj = rest_models.Filter(
                    must=[
                        rest_models.FieldCondition(
                            key="doc_id",
                            match=rest_models.MatchValue(value=doc_id),
                        )
                    ]
                )
            res = self.client.count(
                collection_name=self.collection_name,
                count_filter=filter_obj,
                exact=True,
            )
            return res.count
        except Exception as e:
            logger.debug(f"Count points failed or collection not ready: {e}")
            return 0

vector_store = VectorStoreService()
