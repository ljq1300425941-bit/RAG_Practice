from app.models import DocumentChunk
from app.generator import generate_mock_answer


def test_generate_mock_answer_basic():
    chunks = [
        DocumentChunk("c1", "a.txt", "numpy 支持数值计算", 0, 10),
    ]

    answer = generate_mock_answer("什么是 numpy", chunks)

    assert "什么是 numpy" in answer
    assert "a.txt" in answer
    assert "numpy 支持数值计算" in answer


def test_generate_mock_answer_empty_chunks():
    answer = generate_mock_answer("什么是 numpy", [])

    assert "无法确定" in answer