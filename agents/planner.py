import json
from typing import Any, Dict, List

from llm.ollama_client import OllamaClient


class PlannerAgent:
    TOOL_ALLOWLIST = {"sqlite_query", "python_executor", "web_search"}

    def __init__(self, llm_client: OllamaClient) -> None:
        self.llm_client = llm_client

    def plan(self, query: str) -> Dict[str, Any]:
        prompt = (
            "You are a planner agent. Return only JSON with this exact schema: "
            '{"steps": ["..."], "needs_retrieval": true|false, "tools": ["sqlite_query"|"python_executor"|"web_search"]}. '
            "No markdown. No explanation."
        )
        raw = self.llm_client.generate(prompt=prompt, user_input=query)
        parsed = self._try_parse_json(raw)
        if parsed:
            return self._normalize_plan(parsed, query)
        return self._heuristic_plan(query)

    def _try_parse_json(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            return {}
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}
        return parsed

    def _normalize_plan(self, parsed: Dict[str, Any], query: str) -> Dict[str, Any]:
        steps = parsed.get("steps") or []
        if not isinstance(steps, list):
            steps = []
        steps = [str(step).strip() for step in steps if str(step).strip()]

        raw_tools = parsed.get("tools") or []
        if not isinstance(raw_tools, list):
            raw_tools = []
        tools = [tool for tool in raw_tools if tool in self.TOOL_ALLOWLIST]

        needs_retrieval = bool(parsed.get("needs_retrieval", False))
        if not steps:
            return self._heuristic_plan(query)

        return {
            "steps": steps,
            "needs_retrieval": needs_retrieval,
            "tools": tools,
        }

    def _heuristic_plan(self, query: str) -> Dict[str, Any]:
        q = query.lower()

        retrieval_keywords = ["document", "pdf", "summarize", "summary", "action item", "extract"]
        sql_keywords = ["sql", "database", "sqlite", "table", "query db"]
        python_keywords = ["python", "calculate", "code", "execute"]
        web_keywords = ["web", "internet", "online", "latest", "news", "search"]

        needs_retrieval = any(keyword in q for keyword in retrieval_keywords)
        tools: List[str] = []
        if any(keyword in q for keyword in sql_keywords):
            tools.append("sqlite_query")
        if any(keyword in q for keyword in python_keywords):
            tools.append("python_executor")
        if any(keyword in q for keyword in web_keywords):
            tools.append("web_search")

        steps: List[str] = []
        if needs_retrieval:
            steps.append("retrieve relevant documents")
        if tools:
            steps.append("execute required tools")
        steps.append("compose answer")
        if "action item" in q:
            steps.append("extract action items")
        steps.append("validate answer")

        return {
            "steps": steps,
            "needs_retrieval": needs_retrieval,
            "tools": tools,
        }
