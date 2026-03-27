import re
from typing import Any, Dict, List

from tools.registry import ToolRegistry


class ToolExecutionAgent:
    def __init__(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry

    def execute(self, tools: List[str], query: str) -> List[Dict[str, Any]]:
        tool_results: List[Dict[str, Any]] = []

        for tool_name in dict.fromkeys(tools):
            tool_input: Dict[str, Any] = {"query": query}
            if tool_name == "sqlite_query":
                sql_query = self._extract_sql(query)
                tool_input = {"query": sql_query}
            elif tool_name == "python_executor":
                code = self._extract_python(query)
                tool_input = {"code": code}

            result = self.tool_registry.execute(tool_name, tool_input)
            result["tool"] = tool_name
            tool_results.append(result)

        return tool_results

    def _extract_sql(self, query: str) -> str:
        match = re.search(r"(select\\s+.+)", query, flags=re.IGNORECASE | re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            if not candidate.endswith(";"):
                candidate += ";"
            return candidate
        return "SELECT id, name, owner, status, due_date FROM tasks ORDER BY id LIMIT 5;"

    def _extract_python(self, query: str) -> str:
        fenced = re.search(r"```(?:python)?\\s*(.*?)```", query, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()
        return "print('No python snippet found in user query.')"
