from app.models import DocumentChunk
from app.retriever import build_chunk_embeddings, retrieve_topk_from_embeddings
from app.vectorizer import KeywordCountVectorizer


def test_build_chunk_embeddings():
    chunks = [
        DocumentChunk("c1", "a.txt", "numpy 向量", 0, 10),
        DocumentChunk("c2", "b.txt", "数据库 事务", 0, 10),
    ]
    vectorizer = KeywordCountVectorizer(vocab=["numpy", "向量", "数据库"])

    chunk_embeddings = build_chunk_embeddings(chunks, vectorizer)

    assert len(chunk_embeddings) == 2
    assert chunk_embeddings[0].chunk.chunk_id == "c1"
    assert chunk_embeddings[0].embedding.shape == (3,)


def test_retrieve_topk_from_embeddings():
    chunks = [
        DocumentChunk("c1", "a.txt", "numpy 向量", 0, 10),
        DocumentChunk("c2", "b.txt", "数据库 事务", 0, 10),
        DocumentChunk("c3", "c.txt", "相似度 检索 向量", 0, 10),
    ]
    vectorizer = KeywordCountVectorizer(
        vocab=["numpy", "向量", "相似度", "数据库", "检索"]
    )

    chunk_embeddings = build_chunk_embeddings(chunks, vectorizer)
    results = retrieve_topk_from_embeddings(
        query="numpy 相似度 向量",
        chunk_embeddings=chunk_embeddings,
        vectorizer=vectorizer,
        k=2,
    )

    assert len(results) == 2
    assert results[0][1] >= results[1][1]