import os
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Deque, Dict, Optional

from fastapi import Header, HTTPException


class InMemoryRateLimiter:
    def __init__(self, max_requests: int = 30, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str) -> None:
        now = time.time()
        with self._lock:
            bucket = self._buckets[key]
            while bucket and now - bucket[0] > self.window_seconds:
                bucket.popleft()

            if len(bucket) >= self.max_requests:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

            bucket.append(now)


def parse_rate_limit_from_env(default_per_minute: int = 30) -> int:
    raw = os.getenv("RATE_LIMIT_PER_MINUTE", "").strip()
    if not raw:
        return default_per_minute
    try:
        value = int(raw)
    except ValueError:
        return default_per_minute
    return max(1, value)


def require_optional_bearer(authorization: Optional[str] = Header(default=None)) -> None:
    expected_token = os.getenv("API_BEARER_TOKEN", "").strip()
    if not expected_token:
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.split(" ", 1)[1].strip()
    if token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid bearer token")
