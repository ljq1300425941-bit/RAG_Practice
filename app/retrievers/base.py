from abc import ABC, abstractmethod

from app.schema import RetrievedChunk


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        pass