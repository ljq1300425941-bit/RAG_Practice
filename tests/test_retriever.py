from app.models import DocumentChunk
from app.retriever import retrieve_topk
from app.vectorizer import KeywordCountVectorizer


def test_retrieve_topk_basic():
    chunks = [
        DocumentChunk("c1", "a.txt", "numpy 向量", 0, 10),
        DocumentChunk("c2", "b.txt", "数据库 事务", 0, 10),
        DocumentChunk("c3", "c.txt", "相似度 检索 向量", 0, 10),
    ]

    vectorizer = KeywordCountVectorizer(
        vocab=["numpy", "向量", "相似度", "数据库", "检索"]
    )
    results = retrieve_topk("numpy 相似度 向量", chunks, vectorizer, 2)

    assert len(results) == 2
    assert results[0][1] >= results[1][1]


def test_retrieve_topk_k_too_large():
    chunks = [
        DocumentChunk("c1", "a.txt", "numpy 向量", 0, 10),
    ]
    vectorizer = KeywordCountVectorizer(vocab=["numpy", "向量"])
    results = retrieve_topk("numpy", chunks, vectorizer, 5)

    assert len(results) == 1