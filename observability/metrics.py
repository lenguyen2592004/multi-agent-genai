from threading import Lock
from typing import Any, Dict, List


class MetricsStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._total_requests = 0
        self._successful_requests = 0
        self._latencies_ms: List[int] = []
        self._tool_calls = 0
        self._tool_success = 0
        self._token_usage = 0

    def record_request(self, latency_ms: int, success: bool, token_estimate: int = 0) -> None:
        with self._lock:
            self._total_requests += 1
            if success:
                self._successful_requests += 1
            self._latencies_ms.append(max(0, latency_ms))
            self._token_usage += max(0, token_estimate)

    def record_tool_usage(self, tool_results: List[Dict[str, Any]]) -> None:
        with self._lock:
            for item in tool_results:
                self._tool_calls += 1
                if item.get("status") == "success":
                    self._tool_success += 1

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            avg_latency = (
                float(sum(self._latencies_ms)) / float(len(self._latencies_ms))
                if self._latencies_ms
                else 0.0
            )
            success_rate = (
                float(self._successful_requests) / float(self._total_requests)
                if self._total_requests
                else 0.0
            )
            tool_accuracy = (
                float(self._tool_success) / float(self._tool_calls)
                if self._tool_calls
                else 0.0
            )

            return {
                "total_requests": float(self._total_requests),
                "success_rate": success_rate,
                "avg_latency_ms": avg_latency,
                "tool_accuracy": tool_accuracy,
                "simulated_token_usage": float(self._token_usage),
            }
