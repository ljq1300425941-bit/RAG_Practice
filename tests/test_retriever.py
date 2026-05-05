import numpy as np

from app.models import DocumentChunk, ChunkEmbedding
from app.retriever import retrieve_topk_from_embeddings
from app.schema import RetrievedChunk


class FakeVectorizer:
    def encode(self, text: str):
        if "query" in text:
            return np.array([1.0, 0.0])
        if "match" in text:
            return np.array([1.0, 0.0])
        return np.array([0.0, 1.0])


def test_retrieve_topk_returns_retrieved_chunk():
    chunks = [
        DocumentChunk(
            chunk_id="match_chunk",
            source_file="a.txt",
            text="match text",
            start_pos=0,
            end_pos=10,
        ),
        DocumentChunk(
            chunk_id="other_chunk",
            source_file="b.txt",
            text="other text",
            start_pos=0,
            end_pos=10,
        ),
    ]

    chunk_embeddings = [
        ChunkEmbedding(chunk=chunks[0], embedding=np.array([1.0, 0.0])),
        ChunkEmbedding(chunk=chunks[1], embedding=np.array([0.0, 1.0])),
    ]

    results = retrieve_topk_from_embeddings(
        query="query",
        chunk_embeddings=chunk_embeddings,
        vectorizer=FakeVectorizer(),
        k=1,
    )

    assert len(results) == 1
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].chunk_id == "match_chunk"
    assert results[0].source_file == "a.txt"
    assert results[0].retrieval_score == 1.0