import json
import threading
from pathlib import Path
from typing import List, Optional, Dict
from app.schemas.document import DocumentItem
from app.core.config import settings

class DocumentCatalog:
    """
    Persistent document registry storing metadata of all uploaded/scraped documents.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.catalog_file = settings.upload_dir_path.parent / "documents_catalog.json"
        self._ensure_file()

    def _ensure_file(self):
        if not self.catalog_file.exists():
            self.catalog_file.write_text("{}", encoding="utf-8")

    def _read_data(self) -> Dict[str, dict]:
        try:
            with open(self.catalog_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_data(self, data: Dict[str, dict]):
        with open(self.catalog_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_document(self, item: DocumentItem) -> DocumentItem:
        with self._lock:
            data = self._read_data()
            data[item.doc_id] = item.model_dump()
            self._write_data(data)
            return item

    def get_document(self, doc_id: str) -> Optional[DocumentItem]:
        with self._lock:
            data = self._read_data()
            raw = data.get(doc_id)
            return DocumentItem(**raw) if raw else None

    def list_documents(self) -> List[DocumentItem]:
        with self._lock:
            data = self._read_data()
            return [DocumentItem(**item) for item in data.values()]

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            data = self._read_data()
            if doc_id in data:
                del data[doc_id]
                self._write_data(data)
                return True
            return False

catalog = DocumentCatalog()
