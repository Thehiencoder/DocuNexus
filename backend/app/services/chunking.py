import uuid
from typing import List, Tuple
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from app.schemas.document import ParsedDocument, DocumentChunk
from app.core.config import settings

class ChunkingService:
    """
    Splits ParsedDocuments into semantically coherent chunks using LlamaIndex's
    SentenceSplitter while strictly preserving document metadata, page numbers,
    and character offsets.
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def chunk_document(
        self, doc: ParsedDocument
    ) -> Tuple[List[TextNode], List[DocumentChunk]]:
        """
        Segments a ParsedDocument into chunks per section/page.
        
        Returns:
            Tuple of (List[TextNode] for LlamaIndex/Qdrant, List[DocumentChunk] for API responses).
        """
        nodes: List[TextNode] = []
        chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        # If document has no explicit sections, fallback to raw_text
        sections = doc.sections
        if not sections and doc.raw_text.strip():
            from app.schemas.document import ParsedSection
            sections = [ParsedSection(text=doc.raw_text.strip(), section_title=doc.filename)]

        for section in sections:
            sec_text = section.text.strip()
            if not sec_text:
                continue

            # Split section text using SentenceSplitter
            chunk_texts = self.splitter.split_text(sec_text)
            
            search_start_cursor = 0
            for text_piece in chunk_texts:
                piece_clean = text_piece.strip()
                if not piece_clean:
                    continue

                # Locate character offsets within section
                found_idx = sec_text.find(piece_clean, search_start_cursor)
                if found_idx != -1:
                    char_start = found_idx
                    char_end = found_idx + len(piece_clean)
                    # Advance cursor for overlap
                    search_start_cursor = max(0, found_idx + 1)
                else:
                    char_start = 0
                    char_end = len(piece_clean)

                chunk_id = f"{doc.doc_id}_c{global_chunk_idx}"

                # Rich metadata payload attached to vector index
                metadata = {
                    "chunk_id": chunk_id,
                    "doc_id": doc.doc_id,
                    "filename": doc.filename,
                    "source_type": doc.source_type,
                    "source_url": doc.source_url,
                    "page_number": section.page_number,
                    "section_title": section.section_title,
                    "chunk_index": global_chunk_idx,
                    "char_start": char_start,
                    "char_end": char_end,
                }

                # 1. LlamaIndex TextNode
                node = TextNode(
                    id_=chunk_id,
                    text=piece_clean,
                    metadata=metadata,
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
                nodes.append(node)

                # 2. Pydantic DocumentChunk for API
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    source_type=doc.source_type,
                    source_url=doc.source_url,
                    page_number=section.page_number,
                    section_title=section.section_title,
                    text=piece_clean,
                    char_start=char_start,
                    char_end=char_end,
                    chunk_index=global_chunk_idx,
                    metadata=metadata,
                )
                chunks.append(chunk)

                global_chunk_idx += 1

        return nodes, chunks

# Shared singleton instance
default_chunker = ChunkingService()
