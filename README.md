# DocuNexus 📚⚡

> **Multi-Document Knowledge Base** with Hybrid Search (BM25 + Vector Similarity), Cohere Reranking, and Verifiable Source Attribution.

Built with **LlamaIndex**, **Qdrant**, **Cohere Rerank**, **FastAPI**, and **React**.

---

## 🌟 Key Features

- **Multi-Format Ingestion**: Ingest PDFs (page-level tracking), DOCX, plain text, and web pages (boilerplate stripping).
- **Hybrid Retrieval**: Combines sparse lexical search (BM25) with dense vector similarity search in Qdrant.
- **Precision Reranking**: Utilizes Cohere Rerank (`rerank-v3.5`) to reorder top candidates and eliminate irrelevant noise.
- **Traceable Source Attribution**: Every claim in the LLM answer is cited with interactive badges (`[1]`, `[2]`) that deep-link to the exact source document, page, and snippet.
- **Real-Time Streaming**: Low-latency token streaming via Server-Sent Events (SSE).

---

## 🏗️ Architecture Overview

```text
[Documents / Web Pages]
          │
          ▼
 [Ingestion Pipeline] ──► [Chunking + Metadata Tagging]
                                    │
                                    ▼
                         [Qdrant Collection]
                     (Dense Vector + BM25 Sparse)
                                    │
   [User Query] ────────► [Hybrid Retrieval] (Top 25)
                                    │
                                    ▼
                         [Cohere Rerank v3.5] (Top 5)
                                    │
                                    ▼
                         [Citation Synthesizer]
                                    │
                                    ▼
                   [FastAPI Stream] ──► [React UI]
```

---

## 📁 Repository Structure

```text
DocuNexus/
├── backend/                  # FastAPI backend & LlamaIndex RAG pipeline
│   ├── app/
│   │   ├── api/v1/endpoints/ # API routes (/documents, /query)
│   │   ├── core/             # Configuration & Qdrant client
│   │   ├── services/parsers/ # PDF, DOCX, TXT, and Web parsers
│   │   └── schemas/          # Pydantic data schemas
│   └── storage/uploads/      # Local document storage
├── frontend/                 # React frontend (Vite)
├── .env.example              # Environment variables template
├── .gitignore                # Git exclusions
├── docker-compose.yml        # Qdrant & multi-service setup
└── README.md                 # Project documentation
```

---

## 🚀 Quickstart (Coming soon)
See implementation progress in the execution checklist.
