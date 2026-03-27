import json
from pathlib import Path
from typing import Any, Dict, List

from rag.embeddings import cosine_similarity


class LocalVectorStore:
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
        if not records:
            return
        self._rows.extend(records)
        self._save()

    def count(self) -> int:
        return len(self._rows)

    def similarity_search(self, query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        scored: List[Dict[str, Any]] = []
        for row in self._rows:
            embedding = row.get("embedding", [])
            score = cosine_similarity(query_embedding, embedding)
            scored.append(
                {
                    "id": row.get("id", ""),
                    "text": row.get("text", ""),
                    "metadata": row.get("metadata", {}),
                    "score": score,
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: max(1, top_k)]
