from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Vectorizer(Protocol):
    def encode(self, text: str) -> np.ndarray:
        ...


@dataclass
class KeywordCountVectorizer:
    vocab: list[str]

    def encode(self, text: str) -> np.ndarray:
        text = text.lower()
        values = [text.count(word.lower()) for word in self.vocab]
        return np.array(values, dtype=float)


class EmbeddingVectorizer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        embedding = self.model.encode(text)
        return np.array(embedding, dtype=float)