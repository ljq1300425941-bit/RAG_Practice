import numpy as np

from app.models import DocumentChunk, ChunkEmbedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))

def build_chunk_embeddings(chunks:list[DocumentChunk],vectorizer) -> list[ChunkEmbedding]:
    chunk_embeddings = []

    for chunk in chunks:
        embedding = vectorizer.encode(chunk.text)
        chunk_embeddings.append(
            ChunkEmbedding(
                chunk=chunk,
                embedding=embedding,
            )
        )

    return chunk_embeddings

def retrieve_topk_from_embeddings(
    query: str,
    chunk_embeddings: list[ChunkEmbedding],
    vectorizer,
    k: int,
):
    query_vec = vectorizer.encode(query)
    results = []

    for item in chunk_embeddings:
        score = cosine_similarity(query_vec, item.embedding)
        results.append((item.chunk, score))

    results.sort(key=lambda x: (-x[1], x[0].chunk_id))
    return results[:k]