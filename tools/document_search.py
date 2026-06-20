from typing import Any, Dict

from rag.pipeline import RAGPipeline
from tools.base import BaseTool


class DocumentSearchTool(BaseTool):
    def __init__(self, rag_pipeline: RAGPipeline) -> None:
        self.rag_pipeline = rag_pipeline

    def run(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        query = str(tool_input.get("query", "")).strip()
        top_k = int(tool_input.get("top_k", 4))
        document_id = str(tool_input.get("document_id", "")).strip() or None
        user_id = str(tool_input.get("user_id", "")).strip() or None
        if not query:
            return {"status": "error", "result": "query is required"}

        metadata_filters: Dict[str, Any] = {}
        if document_id:
            metadata_filters["document_id"] = document_id
        docs = self.rag_pipeline.search(
            query=query,
            top_k=top_k,
            metadata_filters=metadata_filters if metadata_filters else None,
            user_id=user_id,
        )
        return {"status": "success", "result": docs}
