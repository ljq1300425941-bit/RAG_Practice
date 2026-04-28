from typing import List

from app.schema import RetrievedChunk
from app.reranker.base import BaseReranker


class NoOpReranker(BaseReranker):
    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_n: int
    ) -> List[RetrievedChunk]:
        return candidates[:top_n]