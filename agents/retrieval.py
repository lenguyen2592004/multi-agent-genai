from typing import Any, Dict, List

from rag.pipeline import RAGPipeline


class RetrievalAgent:
    def __init__(self, rag_pipeline: RAGPipeline) -> None:
        self.rag_pipeline = rag_pipeline

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        return self.rag_pipeline.search(query=query, top_k=top_k)
