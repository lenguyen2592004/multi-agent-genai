from typing import Any, Dict

import requests

from tools.base import BaseTool


class WebSearchTool(BaseTool):
    def __init__(self, timeout_seconds: float = 4.0) -> None:
        self.timeout_seconds = timeout_seconds

    def run(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        query = str(tool_input.get("query", "")).strip()
        if not query:
            return {"status": "error", "result": "query is required"}

        try:
            response = requests.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": query,
                    "format": "json",
                    "no_html": 1,
                    "skip_disambig": 1,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            return {"status": "error", "result": f"Web search unavailable: {exc}"}

        abstract = str(data.get("AbstractText", "")).strip()
        if abstract:
            return {"status": "success", "result": abstract}

        related = data.get("RelatedTopics", [])
        if related and isinstance(related, list):
            for item in related:
                text = str(item.get("Text", "")).strip() if isinstance(item, dict) else ""
                if text:
                    return {"status": "success", "result": text}

        return {"status": "success", "result": "No concise web result found"}
