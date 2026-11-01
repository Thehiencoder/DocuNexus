import logging
from typing import List, Optional
from llama_index.core.schema import TextNode, NodeWithScore
from app.core.config import settings
from app.schemas.query import RetrievedChunk

logger = logging.getLogger(__name__)

class RerankerService:
    """
    Reranks candidate chunks retrieved by Hybrid Search using Cohere Rerank (rerank-v3.5)
    or high-precision fallback, filtering top 25 candidates down to top 4-6 high-confidence chunks.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        top_n: Optional[int] = None,
    ):
        self.api_key = api_key or settings.COHERE_API_KEY
        self.model = model or settings.COHERE_RERANK_MODEL
        self.top_n = top_n or settings.COHERE_TOP_N
        self._cohere_postprocessor = None
        self._init_cohere()

    def _init_cohere(self):
        if self.api_key and self.api_key not in ("your_cohere_api_key_here", ""):
            try:
                from llama_index.postprocessor.cohere_rerank import CohereRerank
                self._cohere_postprocessor = CohereRerank(
                    api_key=self.api_key,
                    top_n=self.top_n,
                    model=self.model,
                )
                logger.info(f"Cohere Rerank ({self.model}) initialized successfully.")
            except Exception as e:
                logger.warning(f"Failed to initialize Cohere Rerank: {e}")
                self._cohere_postprocessor = None
        else:
            logger.info("COHERE_API_KEY not configured. Local fallback reranker will be used.")

    def set_postprocessor(self, postprocessor):
        """Allows injecting a mock or custom postprocessor for testing."""
        self._cohere_postprocessor = postprocessor

    def rerank_nodes(
        self,
        query: str,
        nodes: List[NodeWithScore],
        top_n: Optional[int] = None,
    ) -> List[NodeWithScore]:
        """
        Reranks a list of LlamaIndex NodeWithScore objects using Cohere Rerank.
        """
        if not nodes:
            return []

        target_top_n = top_n or self.top_n

        if self._cohere_postprocessor:
            try:
                self._cohere_postprocessor.top_n = target_top_n
                reranked = self._cohere_postprocessor.postprocess_nodes(
                    nodes=nodes,
                    query_str=query,
                )
                logger.info(f"Cohere reranked {len(nodes)} nodes down to {len(reranked)} nodes.")
                return reranked
            except Exception as e:
                logger.warning(f"Cohere Rerank API call failed, using fallback: {e}")

        # Fallback reranking: rank by term overlap and base score
        return self._fallback_rerank(query, nodes, target_top_n)

    def _fallback_rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        top_n: int,
    ) -> List[NodeWithScore]:
        """
        Deterministic cross-attention heuristic:
        Computes term-overlap density and combines with existing retrieval score.
        """
        query_words = set(query.lower().split())

        def score_node(item: NodeWithScore) -> float:
            text_lower = item.node.get_content().lower()
            text_words = set(text_lower.split())
            if not query_words:
                return float(item.score or 0.0)
            overlap = len(query_words.intersection(text_words)) / len(query_words)
            exact_bonus = 0.5 if query.lower() in text_lower else 0.0
            base_score = float(item.score or 0.0)
            return (base_score * 0.4) + (overlap * 0.4) + (exact_bonus * 0.2)

        scored = [(score_node(n), n) for n in nodes]
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for new_score, item in scored[:top_n]:
            results.append(NodeWithScore(node=item.node, score=round(new_score, 4)))
        return results

    def rerank_chunks(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """
        Reranks a list of RetrievedChunk objects and returns the top-N chunks.
        """
        if not chunks:
            return []

        nodes = [
            NodeWithScore(
                node=TextNode(
                    id_=c.chunk_id,
                    text=c.text,
                    metadata=c.metadata,
                ),
                score=c.score,
            )
            for c in chunks
        ]

        reranked_nodes = self.rerank_nodes(query=query, nodes=nodes, top_n=top_n)

        results = []
        for n in reranked_nodes:
            meta = n.node.metadata or {}
            results.append(
                RetrievedChunk(
                    chunk_id=meta.get("chunk_id", n.node.node_id),
                    doc_id=meta.get("doc_id", ""),
                    filename=meta.get("filename", ""),
                    source_type=meta.get("source_type", ""),
                    source_url=meta.get("source_url"),
                    page_number=meta.get("page_number"),
                    section_title=meta.get("section_title"),
                    text=n.node.get_content(),
                    score=float(n.score) if n.score is not None else 0.0,
                    char_start=meta.get("char_start", 0),
                    char_end=meta.get("char_end", 0),
                    chunk_index=meta.get("chunk_index", 0),
                    metadata=meta,
                )
            )
        return results

reranker_service = RerankerService()
