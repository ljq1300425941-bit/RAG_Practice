from app.models import DocumentChunk
from app.retriever import build_chunk_embeddings, retrieve_topk_from_embeddings
from app.vectorizer import KeywordCountVectorizer
from app.splitter import (
    split_text_to_chunks,
    split_sentences,
)


def test_fixed_splitter_basic():
    chunks = split_text_to_chunks(
        text="abcdefghijklmnopqrstuvwxyz",
        source_file="a.txt",
        chunk_size=10,
        overlap=2,
        splitter_type="fixed",
    )

    assert len(chunks) > 0
    assert chunks[0].chunk_id == "a_chunk_0"
    assert chunks[0].source_file == "a.txt"


def test_split_sentences_chinese():
    sentences = split_sentences("这是第一句。这里是第二句！这是第三句？")

    assert sentences == [
        "这是第一句。",
        "这里是第二句！",
        "这是第三句？",
    ]


def test_sentence_splitter_keeps_sentence_boundary():
    chunks = split_text_to_chunks(
        text="余弦相似度可以衡量向量接近程度。数据库事务保证一致性。",
        source_file="b.txt",
        chunk_size=30,
        overlap=0,
        splitter_type="sentence",
    )

    assert len(chunks) == 2
    assert "余弦相似度" in chunks[0].text
    assert "数据库事务" in chunks[1].text


def test_paragraph_splitter_basic():
    text = "第一段内容。\n\n第二段内容。"

    chunks = split_text_to_chunks(
        text=text,
        source_file="c.txt",
        chunk_size=50,
        overlap=0,
        splitter_type="paragraph",
    )

    assert len(chunks) == 2
    assert chunks[0].text == "第一段内容。"
    assert chunks[1].text == "第二段内容。"


def test_unknown_splitter_raises_error():
    try:
        split_text_to_chunks(
            text="test",
            source_file="a.txt",
            chunk_size=10,
            overlap=0,
            splitter_type="unknown",
        )
        assert False
    except ValueError as e:
        assert "未知 splitter_type" in str(e)

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

    assert results[0].retrieval_score >= results[1].retrieval_score

    assert results[0].chunk_id == "c1"
    assert results[0].source_file == "a.txt"
    assert results[0].text == "numpy 向量"
    assert results[0].rerank_score is None