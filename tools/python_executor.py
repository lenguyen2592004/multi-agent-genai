import subprocess
import sys
from typing import Any, Dict

from tools.base import BaseTool


class PythonExecutorTool(BaseTool):
    BLOCKED_TOKENS = (
        "import os",
        "import sys",
        "subprocess",
        "socket",
        "open(",
        "__import__",
        "eval(",
        "exec(",
    )

    def __init__(self, timeout_seconds: int = 3) -> None:
        self.timeout_seconds = timeout_seconds

    def run(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        code = str(tool_input.get("code", "")).strip()
        if not code:
            return {"status": "error", "result": "Python code is required"}

        lowered = code.lower()
        for blocked in self.BLOCKED_TOKENS:
            if blocked in lowered:
                return {"status": "error", "result": f"Blocked token detected: {blocked}"}

        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"status": "error", "result": "Execution timed out"}

        output = (completed.stdout or "") + (completed.stderr or "")
        output = output.strip()[:2000]
        if completed.returncode != 0:
            return {"status": "error", "result": output or "Python execution failed"}

        return {"status": "success", "result": output or "Execution completed"}
