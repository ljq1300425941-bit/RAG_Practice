import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Simple document retrieval demo")
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
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="embedding 模型名，仅 embedding 模式生效",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="retrieve",
        choices=["retrieve", "rag"],
        help="运行模式：retrieve 或 rag",
    )
    return parser.parse_args()