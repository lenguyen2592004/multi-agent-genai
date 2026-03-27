from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    @abstractmethod
    def run(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
