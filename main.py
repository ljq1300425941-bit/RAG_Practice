from pathlib import Path

from app.cli import parse_args
from app.splitter import build_chunks_from_dir
from app.retriever import build_chunk_embeddings, retrieve_topk_from_embeddings
from app.vectorizer import KeywordCountVectorizer, EmbeddingVectorizer
from app.prompt_builder import build_rag_prompt
from app.generator import generate_answer
from app.reranker import NoOpReranker, CrossEncoderReranker
from app.retrievers import NumpyRetriever, FaissRetriever


def build_retriever(args, chunk_embeddings, vectorizer):
    if args.retriever == "numpy":
        return NumpyRetriever(chunk_embeddings, vectorizer)

    if args.retriever == "faiss":
        return FaissRetriever(chunk_embeddings, vectorizer)

    raise ValueError(f"未知 retriever 类型: {args.retriever}")

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


def print_retrieved_chunks(chunks):
    for rank, chunk in enumerate(chunks, start=1):
        print(f"[Top {rank}]")
        print(f"chunk_id={chunk.chunk_id}")
        print(f"source_file={chunk.source_file}")
        print(f"retrieval_score={chunk.retrieval_score:.4f}")
        print(f"rerank_score={chunk.rerank_score}")
        print(f"text={chunk.text}")
        print("-" * 40)


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    chunks = build_chunks_from_dir(
        input_dir=input_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        splitter_type=args.splitter,
    )

    print(f"共构建 {len(chunks)} 个 chunks")
    print(f"splitter={args.splitter}, chunk_size={args.chunk_size}, overlap={args.overlap}")

    vectorizer = build_vectorizer(args)
    chunk_embeddings = build_chunk_embeddings(chunks, vectorizer)

    print(f"已预计算 {len(chunk_embeddings)} 个 chunk embeddings")
    print("\n开始处理...\n")

    retriever = build_retriever(args, chunk_embeddings, vectorizer)

    candidates = retriever.retrieve(
        query=args.query,
        top_k=args.retrieve_top_k,
    )

    reranker = build_reranker(args)
    retrieved_chunks = reranker.rerank(
        query=args.query,
        candidates=candidates,
        top_n=args.rerank_top_n,
    )

    if args.mode == "retrieve":
        print_retrieved_chunks(retrieved_chunks)

    elif args.mode == "rag":
        prompt = build_rag_prompt(args.query, retrieved_chunks)
        answer = generate_answer(
            query=args.query,
            chunks=retrieved_chunks,
            prompt=prompt,
            generator_type=args.generator,
            llm_model=args.llm_model,
            api_key=args.api_key,
            base_url=args.base_url,
        )

        print("=== Answer ===")
        print(answer)

        print("\n=== References ===")
        for i, chunk in enumerate(retrieved_chunks, start=1):
            print(
                f"[{i}] source={chunk.source_file}, "
                f"chunk_id={chunk.chunk_id}, "
                f"retrieval_score={chunk.retrieval_score:.4f}, "
                f"rerank_score={chunk.rerank_score}"
            )


if __name__ == "__main__":
    main()