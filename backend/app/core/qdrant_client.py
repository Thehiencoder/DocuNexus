import logging
from typing import Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from app.core.config import settings

logger = logging.getLogger(__name__)

def get_qdrant_client() -> QdrantClient:
    """
    Returns an initialized QdrantClient based on settings.
    Supports either local Docker instance (host/port) or Qdrant Cloud (url/api_key).
    """
    if settings.QDRANT_URL:
        return QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=10.0,
        )
    return QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        timeout=10.0,
    )

def init_qdrant_collection(
    client: Optional[QdrantClient] = None,
    collection_name: Optional[str] = None,
    dense_dim: Optional[int] = None,
) -> bool:
    """
    Initializes the Qdrant collection configured with Named Vectors for Hybrid Search:
      - 'dense': Dense float vector (e.g., 768-dim for Gemini text-embedding-004) with Cosine distance.
      - 'sparse': Sparse BM25 vector for exact keyword matches.
    Also ensures payload indexes on 'doc_id', 'source_type', and 'page_number' for fast filtering.
    """
    client = client or get_qdrant_client()
    col_name = collection_name or settings.QDRANT_COLLECTION_NAME
    dim = dense_dim or settings.EMBEDDING_DIM

    try:
        # Check connection
        collections_response = client.get_collections()
        existing_names = [c.name for c in collections_response.collections]

        if col_name not in existing_names:
            logger.info(f"Creating Qdrant collection '{col_name}' with Named Vectors (dense: {dim}d, sparse: BM25)...")
            
            client.create_collection(
                collection_name=col_name,
                vectors_config={
                    "dense": rest_models.VectorParams(
                        size=dim,
                        distance=rest_models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": rest_models.SparseVectorParams(
                        modifier=rest_models.Modifier.IDF,
                    )
                },
            )
            logger.info(f"Collection '{col_name}' created successfully.")
        else:
            logger.info(f"Qdrant collection '{col_name}' already exists.")

        # Create payload indexes for fast filtered lookups and deletions
        try:
            client.create_payload_index(
                collection_name=col_name,
                field_name="doc_id",
                field_schema=rest_models.PayloadSchemaType.KEYWORD,
            )
            client.create_payload_index(
                collection_name=col_name,
                field_name="source_type",
                field_schema=rest_models.PayloadSchemaType.KEYWORD,
            )
            client.create_payload_index(
                collection_name=col_name,
                field_name="page_number",
                field_schema=rest_models.PayloadSchemaType.INTEGER,
            )
        except Exception as e:
            # Index may already exist
            logger.debug(f"Payload index creation note: {e}")

        return True

    except Exception as e:
        logger.warning(f"Could not connect to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}. Detail: {e}")
        return False

def get_collection_status(collection_name: Optional[str] = None) -> Dict[str, Any]:
    """Returns metadata and vector counts for the collection."""
    col_name = collection_name or settings.QDRANT_COLLECTION_NAME
    try:
        client = get_qdrant_client()
        info = client.get_collection(collection_name=col_name)
        return {
            "status": "connected",
            "collection_name": col_name,
            "vectors_count": info.vectors_count or 0,
            "points_count": info.points_count or 0,
            "status_str": str(info.status),
        }
    except Exception as e:
        return {
            "status": "disconnected",
            "collection_name": col_name,
            "error": str(e),
        }
