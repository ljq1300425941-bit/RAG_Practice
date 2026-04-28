import numpy as np
from app.schema import RetrievedChunk
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
) -> list[RetrievedChunk]:
    query_vec = vectorizer.encode(query)
    results: list[RetrievedChunk] = []

    for item in chunk_embeddings:
        score = cosine_similarity(query_vec, item.embedding)

        results.append(
            RetrievedChunk(
                chunk_id=item.chunk.chunk_id,
                text=item.chunk.text,
                source_file=item.chunk.source_file,
                retrieval_score=score,
                rerank_score=None,
            )
        )

    results.sort(key=lambda x: (-x.retrieval_score, x.chunk_id))
    return results[:k]