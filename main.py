from app.retriever import retrieve_topk
from app.splitter import build_chunks_from_dir
from app.cli import parse_args
from pathlib import Path


def main():
    args = parse_args()

    vocab = [
        "python",
        "numpy",
        "向量",
        "相似度",
        "机器学习",
        "数据库",
        "数组",
        "检索",
    ]

    input_dir = Path(args.input_dir)
    chunks = build_chunks_from_dir(
        input_dir = input_dir,
        chunk_size = args.chunk_size,
        overlap = args.overlap,
        )

    print(f"共构建 {len(chunks)} 个 chunks")
    print("\n开始检索...\n")

    results = retrieve_topk(
        query=args.query,
        chunks=chunks,
        vocab=vocab,
        k=args.top_k,
    )

    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"[Top {rank}] score={score:.4f}")
        print(f"chunk_id={chunk.chunk_id}")
        print(f"source_file={chunk.source_file}")
        print(f"text={chunk.text}")
        print("-" * 40)


if __name__ == "__main__":
    main()