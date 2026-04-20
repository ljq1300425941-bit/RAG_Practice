from pathlib import Path

from app.cli import parse_args
from app.splitter import build_chunks_from_dir
from app.retriever import build_chunk_embeddings, retrieve_topk_from_embeddings
from app.vectorizer import KeywordCountVectorizer, EmbeddingVectorizer
from app.prompt_builder import build_rag_prompt
from app.generator import generate_mock_answer


def build_vectorizer(args):
    if args.vectorizer == "keyword":
        vocab = [
            "python",
            "numpy",
            "向量",
            "相似度",
            "机器学习",
            "数据库",
            "数组",
            "检索",
            "事务",
            "数值计算",
        ]
        return KeywordCountVectorizer(vocab=vocab)

    if args.vectorizer == "embedding":
        return EmbeddingVectorizer(model_name=args.model_name)

    raise ValueError(f"未知 vectorizer 类型: {args.vectorizer}")


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    chunks = build_chunks_from_dir(
        input_dir=input_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    print(f"共构建 {len(chunks)} 个 chunks")

    vectorizer = build_vectorizer(args)
    chunk_embeddings = build_chunk_embeddings(chunks, vectorizer)

    print(f"已预计算 {len(chunk_embeddings)} 个 chunk embeddings")
    print("\n开始处理...\n")

    results = retrieve_topk_from_embeddings(
        query=args.query,
        chunk_embeddings=chunk_embeddings,
        vectorizer=vectorizer,
        k=args.top_k,
    )

    retrieved_chunks = [chunk for chunk, score in results]

    if args.mode == "retrieve":
        for rank, (chunk, score) in enumerate(results, start=1):
            print(f"[Top {rank}] score={score:.4f}")
            print(f"chunk_id={chunk.chunk_id}")
            print(f"source_file={chunk.source_file}")
            print(f"text={chunk.text}")
            print("-" * 40)

    elif args.mode == "rag":
        prompt = build_rag_prompt(args.query, retrieved_chunks)
        answer = generate_mock_answer(args.query, retrieved_chunks)

        print("=== Prompt ===")
        print(prompt)
        print("\n=== Answer ===")
        print(answer)

        print("\n=== References ===")
        for i, chunk in enumerate(retrieved_chunks, start=1):
            print(f"[{i}] source={chunk.source_file}, chunk_id={chunk.chunk_id}")


if __name__ == "__main__":
    main()