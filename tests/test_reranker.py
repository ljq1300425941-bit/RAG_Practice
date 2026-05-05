from app.schema import RetrievedChunk
from app.reranker.noop import NoOpReranker


def test_noop_reranker_keeps_original_order():
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            text="text1",
            retrieval_score=0.9,
            source_file="a.txt",
        ),
        RetrievedChunk(
            chunk_id="c2",
            text="text2",
            retrieval_score=0.8,
            source_file="b.txt",
        ),
    ]

    reranker = NoOpReranker()
    results = reranker.rerank(
        query="test query",
        candidates=chunks,
        top_n=1,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert results[0].retrieval_score == 0.9
    assert results[0].rerank_score is None