"""LLM client supporting multiple providers: Ollama, Groq, Gemini, OpenAI."""

import os
import requests
from typing import Optional


class LLMClient:
    """Unified LLM client with fallback support.
    
    Providers (set via LLM_PROVIDER env var):
    - groq: Fast, cheap API (default for cloud)
    - ollama: Local LLM (default for local dev)
    - gemini: Google Gemini API
    - openai: OpenAI API
    """

    def __init__(self) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "ollama").lower().strip()
        self.timeout_seconds = 30.0

    def generate(self, prompt: str, user_input: str) -> str:
        """Generate response using configured provider."""
        generators = {
            "groq": self._groq_generate,
            "gemini": self._gemini_generate,
            "openai": self._openai_generate,
            "ollama": self._ollama_generate,
        }
        
        generator = generators.get(self.provider, self._ollama_generate)
        return generator(prompt, user_input)

    def _call_api(
        self,
        url: str,
        payload: dict,
        headers: dict,
        extract_fn,
        timeout: Optional[float] = None,
    ) -> str:
        """Generic API call with error handling."""
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout or self.timeout_seconds,
            )
            response.raise_for_status()
            return extract_fn(response.json())
        except requests.RequestException:
            return ""
        except (KeyError, IndexError, TypeError):
            return ""

    # ────────────────────────────────────────────
    # Groq API (default for cloud deployment)
    # ────────────────────────────────────────────
    def _groq_generate(self, prompt: str, user_input: str) -> str:
        """Groq API - fast inference, cheap pricing."""
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            return ""

        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        url = "https://api.groq.com/openai/v1/chat/completions"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        def extract(data: dict) -> str:
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
            return ""

        return self._call_api(url, payload, headers, extract)

    # ────────────────────────────────────────────
    # Ollama (local development)
    # ────────────────────────────────────────────
    def _ollama_generate(self, prompt: str, user_input: str) -> str:
        """Ollama local LLM."""
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        full_prompt = f"System:\n{prompt}\n\nUser:\n{user_input}"
        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }

        def extract(data: dict) -> str:
            return str(data.get("response", "")).strip()

        return self._call_api(
            f"{base_url}/api/generate",
            payload,
            {"Content-Type": "application/json"},
            extract,
            timeout=4.0,
        )

    # ────────────────────────────────────────────
    # Gemini (Google)
    # ────────────────────────────────────────────
    def _gemini_generate(self, prompt: str, user_input: str) -> str:
        """Google Gemini API."""
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            return ""

        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"System:\n{prompt}\n\nUser:\n{user_input}"}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 2048,
            }
        }

        def extract(data: dict) -> str:
            candidates = data.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
            return ""

        return self._call_api(url, payload, {"Content-Type": "application/json"}, extract)

    # ────────────────────────────────────────────
    # OpenAI
    # ────────────────────────────────────────────
    def _openai_generate(self, prompt: str, user_input: str) -> str:
        """OpenAI API."""
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return ""

        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        url = "https://api.openai.com/v1/chat/completions"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        def extract(data: dict) -> str:
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
            return ""

        return self._call_api(url, payload, headers, extract)
