import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from tools.base import BaseTool


class SQLiteQueryTool(BaseTool):
    def __init__(self, db_path: Path, row_limit: int = 100) -> None:
        self.db_path = db_path
        self.row_limit = row_limit

    def run(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        query = str(tool_input.get("query", "")).strip()
        if not query:
            return {"status": "error", "result": "SQL query is required"}

        if not re.match(r"^\s*select\b", query, flags=re.IGNORECASE):
            return {"status": "error", "result": "Only SELECT queries are allowed"}

        forbidden = re.search(
            r"\b(drop|delete|update|insert|alter|pragma|attach|detach|create|replace)\b",
            query,
            flags=re.IGNORECASE,
        )
        if forbidden:
            return {"status": "error", "result": f"Forbidden SQL keyword: {forbidden.group(1)}"}

        try:
            with sqlite3.connect(self.db_path) as connection:
                connection.row_factory = sqlite3.Row
                cursor = connection.execute(query)
                rows = cursor.fetchmany(self.row_limit)
        except sqlite3.Error as exc:
            return {"status": "error", "result": str(exc)}

        records: List[Dict[str, Any]] = [dict(row) for row in rows]
        return {"status": "success", "result": records}
