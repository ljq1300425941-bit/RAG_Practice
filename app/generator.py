import os

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


def generate_llm_answer(
    prompt: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("未提供 API key。请通过 --api_key 传入，或设置环境变量 OPENAI_API_KEY。")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("未安装 openai 包，请先执行: pip install openai") from exc

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是一个基于检索上下文回答问题的助手。只能依据提供的上下文回答，若信息不足就明确说无法确定。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


def generate_answer(
    *,
    query: str,
    chunks: list[DocumentChunk],
    prompt: str,
    generator_type: str,
    llm_model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str:
    if generator_type == "mock":
        return generate_mock_answer(query, chunks)

    if generator_type == "llm":
        if not llm_model:
            raise ValueError("generator_type=llm 时必须提供 llm_model。")
        return generate_llm_answer(
            prompt=prompt,
            model=llm_model,
            api_key=api_key,
            base_url=base_url,
        )

    raise ValueError(f"未知 generator_type: {generator_type}")