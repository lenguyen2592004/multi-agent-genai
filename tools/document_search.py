from typing import Any, Dict

from rag.pipeline import RAGPipeline
from tools.base import BaseTool


class DocumentSearchTool(BaseTool):
    def __init__(self, rag_pipeline: RAGPipeline) -> None:
        self.rag_pipeline = rag_pipeline

    def run(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        query = str(tool_input.get("query", "")).strip()
        top_k = int(tool_input.get("top_k", 4))
        if not query:
            return {"status": "error", "result": "query is required"}

        docs = self.rag_pipeline.search(query=query, top_k=top_k)
        return {"status": "success", "result": docs}
