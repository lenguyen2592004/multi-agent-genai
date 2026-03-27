import os

import requests


class OllamaClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.timeout_seconds = 4.0

    def generate(self, prompt: str, user_input: str) -> str:
        full_prompt = f"System:\n{prompt}\n\nUser:\n{user_input}"
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            return ""

        answer = str(data.get("response", "")).strip()
        return answer
