from pathlib import Path
from typing import Any, Dict, List

from rag.pipeline import RAGPipeline
from tools.document_search import DocumentSearchTool
from tools.python_executor import PythonExecutorTool
from tools.sqlite_tool import SQLiteQueryTool
from tools.web_search import WebSearchTool


class ToolRegistry:
    def __init__(self, rag_pipeline: RAGPipeline, db_path: Path) -> None:
        self._tools = {
            "document_search": DocumentSearchTool(rag_pipeline=rag_pipeline),
            "sqlite_query": SQLiteQueryTool(db_path=db_path),
            "python_executor": PythonExecutorTool(),
            "web_search": WebSearchTool(),
        }

    def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        tool = self._tools.get(tool_name)
        if not tool:
            return {"status": "error", "result": f"Unknown tool: {tool_name}"}
        return tool.run(tool_input)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())
