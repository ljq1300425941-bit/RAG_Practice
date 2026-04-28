from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    retrieval_score: float
    source_file: Optional[str] = None
    rerank_score: Optional[float] = None