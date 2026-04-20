from app.models import DocumentChunk


def generate_mock_answer(query: str, chunks: list[DocumentChunk]) -> str:
    if not chunks:
        return "根据当前检索到的资料，无法确定。"

    answer_lines = [
        "这是一个基于检索结果生成的 mock 回答。",
        f"问题：{query}",
        "",
        "检索到的相关资料：",
    ]

    for i, chunk in enumerate(chunks, start=1):
        answer_lines.append(
            f"[{i}] source={chunk.source_file}, chunk_id={chunk.chunk_id}"
        )
        answer_lines.append(chunk.text)
        answer_lines.append("")

    answer_lines.append("当前尚未接入真实生成模型，因此这里只展示基于检索结果组织出的回答草稿。")

    return "\n".join(answer_lines)