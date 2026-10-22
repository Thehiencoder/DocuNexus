import io
import sys
from pathlib import Path

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add backend to sys.path
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

from starlette.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    print("[PASS] Health endpoint verified!")

def test_upload_text_document():
    content = b"# Q3 Financial Performance\nRevenue grew by 24% to $12.5M.\n\n## Expenses\nR&D expenses totaled $3.2M."
    files = {"file": ("report_q3.md", content, "text/markdown")}
    res = client.post("/api/v1/documents/upload", files=files)
    assert res.status_code == 201, f"Failed: {res.text}"
    data = res.json()
    assert data["status"] == "success"
    assert data["filename"] == "report_q3.md"
    assert data["source_type"] == "txt"
    assert data["total_chunks"] >= 1
    doc_id = data["doc_id"]
    print(f"[PASS] Text document upload verified! (doc_id: {doc_id}, chunks: {data['total_chunks']})")
    return doc_id

def test_upload_pdf_document():
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.add_blank_page(width=300, height=300)
    buf = io.BytesIO()
    writer.write(buf)
    pdf_bytes = buf.getvalue()

    files = {"file": ("company_handbook.pdf", pdf_bytes, "application/pdf")}
    res = client.post("/api/v1/documents/upload", files=files)
    assert res.status_code == 201, f"Failed: {res.text}"
    data = res.json()
    assert data["status"] == "success"
    assert data["source_type"] == "pdf"
    assert data["total_pages"] == 2
    print(f"[PASS] PDF document upload verified! (pages: {data['total_pages']})")
    return data["doc_id"]

def test_list_and_raw_retrieval(doc_id: str):
    # List documents
    res = client.get("/api/v1/documents")
    assert res.status_code == 200
    docs = res.json()
    assert any(d["doc_id"] == doc_id for d in docs)
    print(f"[PASS] Document listing verified! ({len(docs)} documents in catalog)")

    # Fetch raw document
    raw_res = client.get(f"/api/v1/documents/{doc_id}/raw")
    assert raw_res.status_code == 200
    assert len(raw_res.content) > 0
    print("[PASS] Raw document retrieval for viewer verified!")

def test_document_deletion(doc_id: str):
    del_res = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200
    print("[PASS] Document deletion verified!")

if __name__ == "__main__":
    print("Testing Ingestion API Endpoints...")
    test_api_health()
    txt_id = test_upload_text_document()
    pdf_id = test_upload_pdf_document()
    test_list_and_raw_retrieval(txt_id)
    test_document_deletion(txt_id)
    test_document_deletion(pdf_id)
    print("ALL API ingestion tests completed successfully!")
