import React, { useState, useEffect } from 'react';
import { 
  Layers, 
  Search, 
  FileText, 
  Sparkles, 
  Server, 
  Database, 
  ExternalLink,
  CheckCircle2,
  Cpu
} from 'lucide-react';
import axios from 'axios';
import './App.css';

export default function App() {
  const [backendStatus, setBackendStatus] = useState({ online: false, data: null });
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const res = await axios.get('http://localhost:8000/health', { timeout: 3000 });
        setBackendStatus({ online: true, data: res.data });
      } catch (err) {
        setBackendStatus({ online: false, data: null });
      } finally {
        setChecking(false);
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-icon">
            <Layers size={22} />
          </div>
          <span className="brand-title">Docu<span className="gradient-text">Nexus</span></span>
          <span className="brand-tag">RAG v1.0</span>
        </div>

        <div className="header-status">
          <div className="status-badge">
            <span className={`status-dot ${backendStatus.online ? 'online' : ''}`}></span>
            <span>API: {backendStatus.online ? 'Online (FastAPI)' : 'Disconnected (port 8000)'}</span>
          </div>
          <div className="status-badge">
            <Database size={14} />
            <span>Qdrant (port 6333)</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="dashboard-content">
        <section className="hero-section">
          <h1 className="hero-title">
            Multi-Document Knowledge Base <br />
            with <span className="gradient-text">Hybrid Search & Citations</span>
          </h1>
          <p className="hero-subtitle">
            Combines dense vector similarity with BM25 lexical retrieval, Cohere cross-encoder reranking, and verified source attribution across PDFs, DOCX, and web pages.
          </p>

          <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
            <a 
              href="http://localhost:8000/docs" 
              target="_blank" 
              rel="noreferrer" 
              className="btn-primary"
            >
              <span>Swagger API Docs</span>
              <ExternalLink size={16} />
            </a>
            <a 
              href="http://localhost:6333/dashboard" 
              target="_blank" 
              rel="noreferrer" 
              className="btn-secondary"
            >
              <span>Qdrant Dashboard</span>
              <ExternalLink size={16} />
            </a>
          </div>
        </section>

        {/* Core Architecture Features */}
        <div className="features-grid">
          <div className="glass-panel feature-card">
            <div className="feature-icon-wrapper">
              <FileText size={22} />
            </div>
            <h3>Multi-Format Ingestion</h3>
            <p>
              Page-aware parser for PDFs, DOCX hierarchies, plain text, and web pages with boilerplate stripping.
            </p>
          </div>

          <div className="glass-panel feature-card">
            <div className="feature-icon-wrapper">
              <Search size={22} />
            </div>
            <h3>Hybrid Search Engine</h3>
            <p>
              Dual-index retrieval combining Google Gemini embeddings (768-dim) with Qdrant BM25 sparse vectors.
            </p>
          </div>

          <div className="glass-panel feature-card">
            <div className="feature-icon-wrapper">
              <Sparkles size={22} />
            </div>
            <h3>Cohere Rerank v3.5</h3>
            <p>
              Cross-attention reranker scoring the top 25 candidates down to 5 high-precision passages for the LLM.
            </p>
          </div>

          <div className="glass-panel feature-card">
            <div className="feature-icon-wrapper">
              <CheckCircle2 size={22} />
            </div>
            <h3>Source Attribution</h3>
            <p>
              Verifiable inline citations linking directly to exact document pages, character ranges, and source URLs.
            </p>
          </div>
        </div>

        {/* Pipeline Progress Overview */}
        <div className="glass-panel pipeline-card">
          <div className="pipeline-header">
            <div>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 700 }}>Project Infrastructure Status</h2>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                Foundation initialized with React, Vite, and FastAPI.
              </p>
            </div>
            <div className="status-badge" style={{ borderColor: 'rgba(16, 185, 129, 0.4)', color: '#34d399' }}>
              <span>Phase 1 Scaffolding Complete</span>
            </div>
          </div>

          <div className="pipeline-steps">
            <div className="pipeline-step-item">
              <div className="step-num">STEP 1.1</div>
              <div className="step-title">Scaffolding</div>
              <div className="step-desc">Repo layout, gitignore, env templates, and structure.</div>
            </div>

            <div className="pipeline-step-item">
              <div className="step-num">STEP 1.2</div>
              <div className="step-title">Qdrant Vector DB</div>
              <div className="step-desc">Docker compose setup for ports 6333 & 6334.</div>
            </div>

            <div className="pipeline-step-item">
              <div className="step-num">STEP 1.3</div>
              <div className="step-title">FastAPI Backend</div>
              <div className="step-desc">Pydantic settings, CORS, and /health endpoint.</div>
            </div>

            <div className="pipeline-step-item" style={{ borderColor: 'var(--accent-primary)' }}>
              <div className="step-num">STEP 1.4</div>
              <div className="step-title">React UI Foundation</div>
              <div className="step-desc">Vite, Lucide icons, Axios, and modern dark design tokens.</div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
