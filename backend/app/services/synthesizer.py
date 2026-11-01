import logging
import re
from typing import List, Tuple, Dict, Optional, Any
from app.core.config import settings
from app.schemas.query import RetrievedChunk

logger = logging.getLogger(__name__)

CITATION_SYSTEM_PROMPT = """You are DocuNexus, an enterprise AI assistant for verified factual question answering across multi-format documents.

Rules you must strictly follow:
1. Rely ONLY on the provided context sources below to answer the user's question. Do not assume or extrapolate facts not directly mentioned in the sources.
2. Every substantive fact, number, date, metric, or claim in your response MUST be explicitly attributed to its source using bracketed numerical citations like [1], [2], or [1][2] immediately following the cited statement.
3. The citation number in brackets corresponds directly to the [Source X] index in the provided context.
4. If the provided context does not contain enough information to answer the question, clearly state: "The provided documents do not contain sufficient information to answer this question."
5. Never fabricate citations or refer to source numbers that were not provided in the context.
"""

class SynthesizerService:
    """
    Constructs citation-constrained prompt templates, formats candidate context nodes
    with numerical source identifiers ([Source 1], [Source 2]), and invokes the LLM.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.LLM_MODEL
        self._llm = None
        self._init_llm()

    def _init_llm(self):
        if self.api_key and self.api_key not in ("your_gemini_api_key_here", ""):
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._llm = genai.GenerativeModel(
                    model_name=self.model,
                    system_instruction=CITATION_SYSTEM_PROMPT,
                )
                logger.info(f"Synthesizer Gemini LLM initialized with model '{self.model}'.")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini GenerativeModel: {e}")
                self._llm = None
        else:
            logger.info("GEMINI_API_KEY not configured. Local fallback synthesizer will be used.")

    def set_llm(self, llm):
        """Allows injecting a mock or custom LLM for testing."""
        self._llm = llm

    def format_sources_context(
        self, chunks: List[RetrievedChunk]
    ) -> Tuple[str, Dict[int, RetrievedChunk]]:
        """
        Formats retrieved chunks into numbered source blocks with rich provenance metadata.
        Returns:
            Tuple of (formatted_context_string, source_map_mapping_1_indexed_int_to_chunk)
        """
        if not chunks:
            return "No relevant sources found in the knowledge base.", {}

        source_blocks: List[str] = []
        source_map: Dict[int, RetrievedChunk] = {}

        for idx, chunk in enumerate(chunks, start=1):
            source_map[idx] = chunk

            # Build provenance header
            provenance_tags = [f"File: {chunk.filename}"]
            if chunk.page_number is not None:
                provenance_tags.append(f"Page: {chunk.page_number}")
            if chunk.section_title and chunk.section_title != chunk.filename:
                provenance_tags.append(f"Section: {chunk.section_title}")
            if chunk.source_url:
                provenance_tags.append(f"URL: {chunk.source_url}")

            header_str = ", ".join(provenance_tags)
            block = f"[Source {idx}] ({header_str}):\n{chunk.text.strip()}"
            source_blocks.append(block)

        context_str = "\n\n".join(source_blocks)
        return context_str, source_map

    def build_citation_prompt(self, query: str, context_str: str) -> str:
        """Constructs the complete user prompt injecting context and question."""
        return f"""Context Information from verified documents:
---------------------
{context_str}
---------------------
Given the context information above and not prior knowledge, answer the following question:
{query}

Remember to cite every fact using [1], [2], etc."""

    def generate_answer(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> Tuple[str, Dict[int, RetrievedChunk]]:
        """
        Generates an answer with inline citations from the retrieved chunks.
        Returns:
            Tuple of (answer_text, source_map)
        """
        context_str, source_map = self.format_sources_context(chunks)
        if not source_map:
            return "The provided documents do not contain sufficient information to answer this question.", {}

        prompt = self.build_citation_prompt(query, context_str)

        # 1. Use real Gemini LLM if available
        if self._llm is not None:
            try:
                response = self._llm.generate_content(prompt)
                if response and response.text:
                    return response.text.strip(), source_map
            except Exception as e:
                logger.warning(f"Gemini LLM generation failed, falling back to local synthesizer: {e}")

        # 2. Local fallback synthesizer for offline verification/testing
        answer = self._fallback_synthesize(query, chunks, source_map)
        return answer, source_map

    def _fallback_synthesize(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        source_map: Dict[int, RetrievedChunk],
    ) -> str:
        """
        Deterministic offline synthesizer that extracts matching sentences from the top sources
        and attaches strictly valid inline citations [1], [2].
        """
        query_words = set(query.lower().split())
        extracted_facts: List[str] = []

        for idx, chunk in source_map.items():
            sentences = re.split(r'(?<=[.!?])\s+', chunk.text.strip())
            best_sentence = None
            best_overlap = 0

            for s in sentences:
                s_words = set(s.lower().split())
                overlap = len(query_words.intersection(s_words))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sentence = s.strip()

            if best_sentence:
                # Append inline citation tag [X]
                clean_s = best_sentence.rstrip(".")
                extracted_facts.append(f"{clean_s} [{idx}].")

        if extracted_facts:
            return " ".join(extracted_facts)

        # Fallback to the first source with citation
        first_chunk_text = chunks[0].text.strip().rstrip(".")
        return f"Based on the available documents, {first_chunk_text.lower()} [1]."

synthesizer_service = SynthesizerService()
