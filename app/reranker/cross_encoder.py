from dataclasses import replace
from typing import List

from app.schema import RetrievedChunk
from app.reranker.base import BaseReranker


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "未安装 sentence-transformers，请先执行: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_n: int
    ) -> List[RetrievedChunk]:
        if not candidates:
            return []

        pairs = [(query, chunk.text) for chunk in candidates]
        scores = self.model.predict(pairs)

        reranked_chunks: list[RetrievedChunk] = []

        for chunk, score in zip(candidates, scores):
            reranked_chunks.append(
                replace(
                    chunk,
                    rerank_score=float(score),
                )
            )

        reranked_chunks.sort(
            key=lambda x: (
                -(x.rerank_score if x.rerank_score is not None else float("-inf")),
                -x.retrieval_score,
                x.chunk_id,
            )
        )

        return reranked_chunks[:top_n]