from app.schema import RetrievedChunk
from evaluate import top1_hit, recall_at_k, reciprocal_rank


def make_chunk(chunk_id: str):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text="test",
        retrieval_score=1.0,
        source_file="test.txt",
    )


def test_top1_hit():
    results = [make_chunk("a"), make_chunk("b")]
    assert top1_hit(results, {"a"}) == 1
    assert top1_hit(results, {"b"}) == 0


def test_recall_at_k():
    results = [make_chunk("a"), make_chunk("b"), make_chunk("c")]
    assert recall_at_k(results, {"c"}, 3) == 1
    assert recall_at_k(results, {"c"}, 2) == 0


def test_reciprocal_rank():
    results = [make_chunk("a"), make_chunk("b"), make_chunk("c")]
    assert reciprocal_rank(results, {"a"}) == 1.0
    assert reciprocal_rank(results, {"b"}) == 0.5
    assert reciprocal_rank(results, {"x"}) == 0.0