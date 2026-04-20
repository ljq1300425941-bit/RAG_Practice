from app.models import DocumentChunk
from app.prompt_builder import build_rag_prompt


def test_build_rag_prompt_basic():
    chunks = [
        DocumentChunk("c1", "a.txt", "numpy 支持数值计算", 0, 10),
        DocumentChunk("c2", "b.txt", "余弦相似度用于衡量向量接近程度", 0, 10),
    ]

    prompt = build_rag_prompt("向量接近程度怎么衡量", chunks)

    assert "向量接近程度怎么衡量" in prompt
    assert "numpy 支持数值计算" in prompt
    assert "余弦相似度用于衡量向量接近程度" in prompt
    assert "source=a.txt" in prompt
    assert "chunk_id=c2" in prompt