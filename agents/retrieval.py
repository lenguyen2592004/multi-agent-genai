from typing import Any, Dict, List, Optional

from rag.pipeline import RAGPipeline


class RetrievalAgent:
    def __init__(self, rag_pipeline: RAGPipeline) -> None:
        self.rag_pipeline = rag_pipeline

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        document_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        metadata_filters: Dict[str, Any] = {}
        if document_id:
            metadata_filters["document_id"] = document_id
        return self.rag_pipeline.search(
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters,
            user_id=user_id,
        )
