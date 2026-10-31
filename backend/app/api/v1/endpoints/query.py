from fastapi import APIRouter
from app.schemas.query import HybridSearchRequest, HybridSearchResponse
from app.services.retriever import hybrid_retriever

router = APIRouter(prefix="/query", tags=["Query & Retrieval"])

@router.post("/search", response_model=HybridSearchResponse)
async def hybrid_search(payload: HybridSearchRequest):
    """
    Executes hybrid retrieval combining dense semantic vectors (Gemini 768d)
    and sparse lexical vectors (FastEmbed BM25) via Reciprocal Rank Fusion (RRF).
    Returns top-K candidate chunks with relevance scores and metadata.
    """
    results = hybrid_retriever.search(
        query=payload.query,
        top_k=payload.top_k,
        doc_ids=payload.doc_ids,
    )
    return HybridSearchResponse(
        query=payload.query,
        total_results=len(results),
        results=results,
    )
