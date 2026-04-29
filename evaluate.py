import argparse
import json
import time
from pathlib import Path

from app.splitter import build_chunks_from_dir
from app.retriever import build_chunk_embeddings, retrieve_topk_from_embeddings
from app.vectorizer import KeywordCountVectorizer, EmbeddingVectorizer
from app.reranker import NoOpReranker, CrossEncoderReranker


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


def build_reranker(args):
    if args.reranker == "none":
        return NoOpReranker()

    if args.reranker == "cross_encoder":
        return CrossEncoderReranker(model_name=args.reranker_model)

    raise ValueError(f"未知 reranker 类型: {args.reranker}")


def load_questions(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def top1_hit(results, expected_ids: set[str]) -> int:
    if not results:
        return 0
    return 1 if results[0].chunk_id in expected_ids else 0


def recall_at_k(results, expected_ids: set[str], k: int) -> int:
    top_k_ids = {chunk.chunk_id for chunk in results[:k]}
    return 1 if top_k_ids & expected_ids else 0


def reciprocal_rank(results, expected_ids: set[str]) -> float:
    for index, chunk in enumerate(results, start=1):
        if chunk.chunk_id in expected_ids:
            return 1.0 / index
    return 0.0


def evaluate(args):
    questions = load_questions(Path(args.eval_file))

    chunks = build_chunks_from_dir(
        input_dir=Path(args.input_dir),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    print(f"共构建 {len(chunks)} 个 chunks")

    vectorizer = build_vectorizer(args)
    chunk_embeddings = build_chunk_embeddings(chunks, vectorizer)

    print(f"已预计算 {len(chunk_embeddings)} 个 chunk embeddings")

    reranker = build_reranker(args)

    total = len(questions)
    top1_hits = 0
    recall_hits = 0
    mrr_sum = 0.0
    latency_sum = 0.0

    print("\n开始评估...\n")

    for item in questions:
        query = item["query"]
        expected_ids = set(item["expected_chunk_ids"])

        start_time = time.perf_counter()

        candidates = retrieve_topk_from_embeddings(
            query=query,
            chunk_embeddings=chunk_embeddings,
            vectorizer=vectorizer,
            k=args.retrieve_top_k,
        )

        results = reranker.rerank(
            query=query,
            candidates=candidates,
            top_n=args.rerank_top_n,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        latency_sum += elapsed_ms

        q_top1_hit = top1_hit(results, expected_ids)
        q_recall = recall_at_k(results, expected_ids, args.rerank_top_n)
        q_rr = reciprocal_rank(results, expected_ids)

        top1_hits += q_top1_hit
        recall_hits += q_recall
        mrr_sum += q_rr

        print(f"Query: {query}")
        print(f"Expected: {sorted(expected_ids)}")
        print(f"Top results:")
        for rank, chunk in enumerate(results, start=1):
            print(
                f"  [{rank}] chunk_id={chunk.chunk_id}, "
                f"retrieval_score={chunk.retrieval_score:.4f}, "
                f"rerank_score={chunk.rerank_score}"
            )
        print(
            f"Top1Hit={q_top1_hit}, "
            f"Recall@{args.rerank_top_n}={q_recall}, "
            f"RR={q_rr:.4f}, "
            f"Latency={elapsed_ms:.2f}ms"
        )
        print("-" * 80)

    print("\n=== Summary ===")
    print(f"vectorizer={args.vectorizer}")
    print(f"reranker={args.reranker}")
    print(f"retrieve_top_k={args.retrieve_top_k}")
    print(f"rerank_top_n={args.rerank_top_n}")
    print(f"Top1 Hit Rate={top1_hits / total:.4f}")
    print(f"Recall@{args.rerank_top_n}={recall_hits / total:.4f}")
    print(f"MRR={mrr_sum / total:.4f}")
    print(f"Avg Latency={latency_sum / total:.2f}ms")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval / rerank quality")

    parser.add_argument("--input_dir", type=str, default="data")
    parser.add_argument("--eval_file", type=str, default="eval/questions.json")
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--overlap", type=int, default=5)

    parser.add_argument(
        "--vectorizer",
        type=str,
        default="keyword",
        choices=["keyword", "embedding"],
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    parser.add_argument(
        "--reranker",
        type=str,
        default="none",
        choices=["none", "cross_encoder"],
    )
    parser.add_argument(
        "--reranker_model",
        type=str,
        default="BAAI/bge-reranker-base",
    )

    parser.add_argument("--retrieve_top_k", type=int, default=5)
    parser.add_argument("--rerank_top_n", type=int, default=3)

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())