import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.embeddings import cosine_similarity


class LocalVectorStore:
    """JSON-backed vector store with document/user isolation.

    Each record tracks ``document_id`` and ``user_id`` so callers can filter
    results to a single user's uploaded documents and deprioritise stale
    content.
    """

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.file_path.exists():
            self._rows = []
            return
        try:
            raw = json.loads(self.file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._rows = []
            return
        if isinstance(raw, list):
            self._rows = [row for row in raw if isinstance(row, dict)]
        else:
            self._rows = []

    def _save(self) -> None:
        self.file_path.write_text(json.dumps(self._rows, ensure_ascii=True, indent=2), encoding="utf-8")

    def add(self, records: List[Dict[str, Any]]) -> None:
        """Append records to the store."""
        if not records:
            return
        for record in records:
            record.setdefault("created_at", time.time())
        self._rows.extend(records)
        self._save()

    def delete_by_document_id(self, document_id: str) -> int:
        """Remove all chunks belonging to ``document_id``.

        Returns the number of removed rows.
        """
        original_len = len(self._rows)
        self._rows = [
            row for row in self._rows
            if row.get("metadata", {}).get("document_id") != document_id
        ]
        removed = original_len - len(self._rows)
        if removed:
            self._save()
        return removed

    def delete_by_user_id(self, user_id: str) -> int:
        """Remove all chunks belonging to ``user_id``."""
        original_len = len(self._rows)
        self._rows = [
            row for row in self._rows
            if row.get("metadata", {}).get("user_id") != user_id
        ]
        removed = original_len - len(self._rows)
        if removed:
            self._save()
        return removed

    def count(self) -> int:
        return len(self._rows)

    def list_documents(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return unique documents, optionally filtered by user."""
        seen: set = set()
        documents: List[Dict[str, Any]] = []
        for row in self._rows:
            metadata = row.get("metadata", {})
            if user_id is not None and metadata.get("user_id") != user_id:
                continue
            doc_id = metadata.get("document_id")
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                documents.append({
                    "document_id": doc_id,
                    "metadata": metadata,
                    "created_at": row.get("created_at", 0),
                })
        return documents

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return the most similar chunks.

        Results are ranked by cosine similarity.  When multiple chunks have
        identical similarity, newer chunks (``created_at``) are preferred.
        """
        scored: List[Dict[str, Any]] = []
        for row in self._rows:
            metadata = row.get("metadata", {})
            if metadata_filters:
                should_skip = False
                for key, expected_value in metadata_filters.items():
                    if expected_value is None:
                        continue
                    if metadata.get(key) != expected_value:
                        should_skip = True
                        break
                if should_skip:
                    continue

            embedding = row.get("embedding", [])
            score = cosine_similarity(query_embedding, embedding)
            scored.append(
                {
                    "id": row.get("id", ""),
                    "text": row.get("text", ""),
                    "metadata": metadata,
                    "score": score,
                    "created_at": row.get("created_at", 0),
                }
            )
        # Sort by score desc, then by recency desc as a tie-breaker.
        scored.sort(key=lambda item: (item["score"], item["created_at"]), reverse=True)
        return scored[: max(1, top_k)]
