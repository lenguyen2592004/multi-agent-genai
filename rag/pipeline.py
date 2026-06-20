from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.chunker import chunk_text
from rag.embeddings import embed_text
from rag.vector_store import LocalVectorStore


class RAGPipeline:
    """End-to-end RAG pipeline: chunk -> embed -> store -> search.

    Notes
    -----
    * Uploading a document with the same ``document_id`` replaces the previous
      version instead of appending duplicate chunks.
    * ``ingest_directory`` is intended for one-time bootstrapping from a local
      ``docs/`` folder; it skips files whose ``document_id`` already exists.
    """

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
        """Ingest text into the vector store.

        If ``document_id`` already exists, the old chunks are removed first so
        the store never returns stale data for the same document.
        """
        # Replace existing document instead of duplicating chunks.
        self.vector_store.delete_by_document_id(document_id)

        chunks = chunk_text(text=text, chunk_size=self.chunk_size, overlap=self.overlap)
        records: List[Dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            records.append(
                {
                    "id": f"{document_id}::chunk-{index}",
                    "text": chunk,
                    "metadata": {
                        **metadata,
                        "document_id": document_id,
                        "chunk_index": index,
                    },
                    "embedding": embed_text(chunk),
                }
            )
        self.vector_store.add(records)
        return len(records)

    def ingest_directory(self, directory: Path) -> int:
        """Ingest .txt/.md files from ``directory`` once per boot.

        Files whose ``document_id`` (file stem) already exists are skipped so
        the pipeline does not resurrect stale content across restarts.
        """
        if not directory.exists():
            return 0

        existing_documents = {
            doc["document_id"]
            for doc in self.vector_store.list_documents()
        }

        count = 0
        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in {".txt", ".md"}:
                continue
            document_id = file_path.stem
            if document_id in existing_documents:
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except OSError:
                continue
            if not text.strip():
                continue
            count += self.ingest_text(
                document_id=document_id,
                text=text,
                metadata={"source": str(file_path.name), "type": "bootstrap"},
            )
        return count

    def search(
        self,
        query: str,
        top_k: int = 4,
        metadata_filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search the vector store.

        If ``user_id`` is provided, only that user's uploaded documents are
        searched.  This prevents one user's uploads from polluting another
        user's results.
        """
        filters: Dict[str, Any] = dict(metadata_filters or {})
        if user_id is not None:
            filters["user_id"] = user_id

        query_embedding = embed_text(query)
        return self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filters=filters,
        )

    def clear_user_documents(self, user_id: str) -> int:
        """Remove all documents belonging to ``user_id``."""
        return self.vector_store.delete_by_user_id(user_id)

    def list_user_documents(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return documents for a user (or all documents if omitted)."""
        return self.vector_store.list_documents(user_id=user_id)
