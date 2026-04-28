from app.schema import RetrievedChunk


def build_rag_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[{i}] source={chunk.source_file}, chunk_id={chunk.chunk_id}\n{chunk.text}"
        )

    context_text = "\n\n".join(context_parts) if context_parts else "无可用上下文"

    prompt = f"""你是一个基于给定资料回答问题的助手。
请严格根据提供的上下文回答，不要使用上下文之外的知识。
如果上下文不足以回答问题，就明确说“根据当前检索到的资料，无法确定”。

问题：
{query}

上下文：
{context_text}

请输出：
1. 一个简洁回答
2. 如果可以，简要说明依据
3. 不要编造上下文中没有的信息
"""
    return prompt