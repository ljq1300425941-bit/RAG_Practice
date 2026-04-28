import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Simple document retrieval / RAG demo")
    parser.add_argument("--input_dir", type=str, required=True, help="输入文档目录")
    parser.add_argument("--query", type=str, required=True, help="检索查询")
    parser.add_argument("--chunk_size", type=int, default=30, help="chunk 大小")
    parser.add_argument("--overlap", type=int, default=5, help="chunk 重叠长度")
    parser.add_argument("--top_k", type=int, default=3, help="返回前 k 个结果")

    parser.add_argument(
        "--vectorizer",
        type=str,
        default="keyword",
        choices=["keyword", "embedding"],
        help="向量化方式：keyword 或 embedding",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="embedding 模型名，仅 embedding 模式生效",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="retrieve",
        choices=["retrieve", "rag"],
        help="运行模式：retrieve 或 rag",
    )

    parser.add_argument(
        "--generator",
        type=str,
        default="mock",
        choices=["mock", "llm"],
        help="回答生成方式：mock 或 llm",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="gpt-4.1-mini",
        help="真实生成模型名，仅 llm generator 生效",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="LLM API key，可选；未提供时也可从环境变量读取",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="可选，自定义 LLM 接口地址",
    )

    parser.add_argument(
        "--reranker",
        type=str,
        default="none",
        choices=["none", "cross_encoder"],
        help="Reranker type. Currently supports: none"
    )

    parser.add_argument(
        "--rerank_top_n",
        type=int,
        default=5,
        help="Number of chunks kept after reranking"
    )

    parser.add_argument(
        "--retrieve_top_k",
        type=int,
        default=30,
        help="Number of candidate chunks retrieved before reranking"
    )

    parser.add_argument(
        "--reranker_model",
        type=str,
        default="BAAI/bge-reranker-base",
        help="reranker 模型名，仅 cross_encoder reranker 生效"
    )

    return parser.parse_args()