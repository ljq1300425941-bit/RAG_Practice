from app.models import ChunkEmbedding
from app.retriever import retrieve_topk_from_embeddings
from app.retrievers.base import BaseRetriever
from app.schema import RetrievedChunk


class NumpyRetriever(BaseRetriever):
    def __init__(self, chunk_embeddings: list[ChunkEmbedding], vectorizer):
        self.chunk_embeddings = chunk_embeddings
        self.vectorizer = vectorizer

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return retrieve_topk_from_embeddings(
            query=query,
            chunk_embeddings=self.chunk_embeddings,
            vectorizer=self.vectorizer,
            k=top_k,
        )