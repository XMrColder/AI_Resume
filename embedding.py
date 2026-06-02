from functools import lru_cache
from typing import List, Tuple

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


@lru_cache(maxsize=4096)
def _encode(text: str):
    return _get_model().encode(text, normalize_embeddings=True)


def similarity(a: str, b: str) -> float:
    import numpy as np
    return float(np.dot(_encode(a), _encode(b)))


def best_match(query: str, candidates: List[str]) -> Tuple[int, float]:
    if not candidates:
        return -1, 0.0
    scores = [similarity(query, c) for c in candidates]
    i = max(range(len(scores)), key=lambda k: scores[k])
    return i, scores[i]
