import hashlib
import math
from typing import List


def embed_text(text: str, dim: int = 128) -> List[float]:
    vector = [0.0 for _ in range(dim)]
    tokens = text.lower().split()
    if not tokens:
        return vector

    for token in tokens:
        index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim
        vector[index] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b))
