from app.models import DocumentChunk
from app.vectorizer import text_to_vector
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def retrieve_topk(query: str, chunks: list[DocumentChunk], vocab: list[str], k: int):
    query_vec = text_to_vector(query, vocab)
    results = []

    for chunk in chunks:
        chunk_vec = text_to_vector(chunk.text, vocab)
        score = cosine_similarity(query_vec, chunk_vec)
        results.append((chunk, score))

    results.sort(key=lambda x: (-x[1], x[0].chunk_id))
    return results[:k]