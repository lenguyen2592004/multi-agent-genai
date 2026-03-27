from pathlib import Path
from typing import Any, Dict, List

from rag.chunker import chunk_text
from rag.embeddings import embed_text
from rag.vector_store import LocalVectorStore


class RAGPipeline:
    def __init__(
        self,
        vector_store_path: Path,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> None:
        self.vector_store = LocalVectorStore(vector_store_path)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def document_count(self) -> int:
        return self.vector_store.count()

    def ingest_text(self, document_id: str, text: str, metadata: Dict[str, Any]) -> int:
        chunks = chunk_text(text=text, chunk_size=self.chunk_size, overlap=self.overlap)
        records: List[Dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            records.append(
                {
                    "id": f"{document_id}::chunk-{index}",
                    "text": chunk,
                    "metadata": {**metadata, "document_id": document_id, "chunk_index": index},
                    "embedding": embed_text(chunk),
                }
            )
        self.vector_store.add(records)
        return len(records)

    def ingest_directory(self, directory: Path) -> int:
        if not directory.exists():
            return 0
        count = 0
        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in {".txt", ".md"}:
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except OSError:
                continue
            if not text.strip():
                continue
            count += self.ingest_text(
                document_id=file_path.stem,
                text=text,
                metadata={"source": str(file_path.name)},
            )
        return count

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        query_embedding = embed_text(query)
        return self.vector_store.similarity_search(query_embedding=query_embedding, top_k=top_k)
