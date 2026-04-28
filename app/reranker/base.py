from abc import ABC, abstractmethod
from typing import List

from app.schema import RetrievedChunk


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_n: int
    ) -> List[RetrievedChunk]:
        pass