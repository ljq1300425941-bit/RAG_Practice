import numpy as np

from app.models import ChunkEmbedding
from app.retrievers.base import BaseRetriever
from app.schema import RetrievedChunk


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


class FaissRetriever(BaseRetriever):
    def __init__(self, chunk_embeddings: list[ChunkEmbedding], vectorizer):
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("未安装 faiss，请先执行: pip install faiss-cpu") from exc

        self.faiss = faiss
        self.chunk_embeddings = chunk_embeddings
        self.vectorizer = vectorizer

        if not chunk_embeddings:
            self.index = None
            return

        vectors = np.vstack(
            [item.embedding for item in chunk_embeddings]
        ).astype("float32")

        vectors = _normalize_vectors(vectors)

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self.index is None:
            return []

        query_vec = self.vectorizer.encode(query).astype("float32")
        query_vec = query_vec.reshape(1, -1)
        query_vec = _normalize_vectors(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        results: list[RetrievedChunk] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            chunk = self.chunk_embeddings[int(idx)].chunk

            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source_file=chunk.source_file,
                    retrieval_score=float(score),
                    rerank_score=None,
                )
            )

        return results