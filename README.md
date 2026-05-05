# 简易文档检索 / RAG 原型（README V4.0）

一个基于 Python 实现的轻量级文档检索 / RAG 原型。项目从底层实现了文档读取、chunk 切分、向量化、top-k 召回、二阶段 rerank、prompt 构造、mock / llm generator、检索评估、chunking 策略对比，以及 numpy / FAISS 两种向量检索后端。

当前版本重点不是堆叠 LangChain、向量数据库或复杂 Agent，而是把 RAG 主链路做成一个可运行、可解释、可评估、可扩展的小型工程项目。

```text
文档读取
  → chunk 切分 fixed / paragraph / sentence
  → metadata 管理
  → keyword / embedding 向量化
  → chunk embedding 预计算
  → numpy / FAISS 检索后端
  → 第一阶段 top-k 召回
  → 第二阶段 rerank 重排序
  → prompt 构造
  → mock / llm generator
  → answer + references 输出
  → evaluation + error analysis
```

---

## 1. 项目定位

这是一个用于学习和验证 RAG 核心链路的轻量项目，目标是回答几个关键问题：

1. 文档如何切分成适合检索的 chunk？
2. keyword retrieval 和 embedding retrieval 有什么差异？
3. 为什么第一阶段召回之后还需要 rerank？
4. 如何用指标评估检索效果，而不是只靠人工观察？
5. numpy 暴力检索如何扩展到 FAISS 向量索引？
6. RAG 项目中哪些能力是真正影响效果的核心环节？

因此，本项目没有直接使用 LangChain 封装主流程，而是手动实现 splitter、vectorizer、retriever、reranker、prompt builder、generator 和 evaluator，便于理解每个模块的作用和工程取舍。

---

## 2. 当前功能

当前 V4.0 支持：

- 读取指定目录下的多个 `.txt` 文件；
- 为文本构建 `DocumentChunk` metadata；
- 支持三种 chunking 策略：
  - `fixed`：固定字符长度切分，支持 overlap；
  - `paragraph`：按段落切分，尽量保留自然语义边界；
  - `sentence`：按句子切分，并在 chunk size 内合并短句；
- 支持两种向量化方式：
  - `keyword`：关键词计数向量，作为可解释 baseline；
  - `embedding`：基于 sentence-transformers 的真实 embedding；
- 对 chunk embeddings 做预计算，避免查询时重复编码全部文档；
- 支持两种 retriever 后端：
  - `numpy`：本地暴力余弦相似度检索；
  - `faiss`：基于 FAISS `IndexFlatIP` 的向量索引检索；
- 支持两种 reranker：
  - `none`：不重排，作为 baseline；
  - `cross_encoder`：使用 CrossEncoder / BGE reranker 对候选 chunk 重新打分；
- 支持两种运行模式：
  - `retrieve`：仅输出检索 / rerank 结果；
  - `rag`：检索后构造 prompt 并生成回答；
- 支持两种生成方式：
  - `mock`：基于检索结果输出回答草稿；
  - `llm`：预留真实 LLM generator 接口；
- 支持检索评估：
  - `Top1 Hit Rate`
  - `Recall@K`
  - `MRR`
  - `Avg Latency`
- 支持基于 `expected_chunk_ids` 或 `expected_keywords` 的评估方式；
- 支持 error analysis，记录成功案例和失败案例；
- 已补充基础测试，保证 splitter、retriever、reranker、evaluator 等核心模块稳定。

---

## 3. 项目结构

```text
RAG_Practice/
  main.py
  evaluate.py
  app/
    __init__.py
    cli.py
    models.py
    schema.py
    loader.py
    splitter.py
    vectorizer.py
    retriever.py
    prompt_builder.py
    generator.py
    reranker/
      __init__.py
      base.py
      noop.py
      cross_encoder.py
    retrievers/
      __init__.py
      base.py
      numpy_retriever.py
      faiss_retriever.py
  data/
    a.txt
    b.txt
    c.txt
  eval/
    questions.json
    error_cases.md
  tests/
    test_splitter.py
    test_retriever.py
    test_reranker.py
    test_evaluator.py
    test_prompt_builder.py
    test_generator.py
    test_faiss_retriever.py
  requirements.txt
  README.md
```

其中：

- `app/splitter.py`：负责 fixed / paragraph / sentence 三种切分策略；
- `app/vectorizer.py`：负责 keyword / embedding 向量化；
- `app/retrievers/`：封装 numpy / FAISS 两种检索后端；
- `app/reranker/`：封装 NoOp / CrossEncoder 两种 rerank 策略；
- `evaluate.py`：检索效果评估脚本；
- `eval/questions.json`：人工标注的评估 query；
- `eval/error_cases.md`：成功案例与失败案例分析。

---

## 4. 核心模块说明

### 4.1 `models.py`

定义文档和向量相关的基础结构：

- `DocumentChunk`
- `ChunkEmbedding`

`DocumentChunk` 表示切分后的文本块，包含：

- `chunk_id`
- `source_file`
- `text`
- `start_pos`
- `end_pos`

`ChunkEmbedding` 表示一个 chunk 及其对应的 embedding。

---

### 4.2 `schema.py`

定义检索阶段统一传递的数据结构：

```python
@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    retrieval_score: float
    source_file: Optional[str] = None
    rerank_score: Optional[float] = None
```

引入 `RetrievedChunk` 后，retriever 和 reranker 之间不再传递 `(chunk, score)` tuple，而是传递结构化对象。

这样做的好处是：

- 同时保留 `retrieval_score` 和 `rerank_score`；
- prompt builder / generator 可以直接复用 rerank 后的结果；
- 后续扩展 FAISS、Qdrant、BM25、hybrid search 时接口更稳定；
- 输出结果更适合调试和实验分析。

---

### 4.3 `splitter.py`

负责文本切分。

当前支持三种策略。

#### fixed splitter

固定字符长度切分，支持 overlap。

优点：

- 实现简单；
- 行为稳定；
- 适合作为 baseline。

缺点：

- 容易切断句子；
- 容易破坏语义边界；
- 可能导致正确答案依据分散到不同 chunk 中。

#### paragraph splitter

按段落切分，尽量保留自然文档边界。

适合：

- 笔记；
- README；
- 教程文档；
- 段落结构较清晰的知识库。

#### sentence splitter

按句子切分，并在 `chunk_size` 限制内合并多个短句。

适合：

- 中文短文本；
- 说明性文档；
- 需要尽量避免句子被截断的场景。

---

### 4.4 `vectorizer.py`

负责文本转向量。

#### `KeywordCountVectorizer`

一个简化的 bag-of-words baseline，根据预定义词表统计关键词出现次数。

优点：

- 简单；
- 可解释；
- 适合做 baseline；
- 在强关键词匹配场景下表现稳定。

局限：

- 依赖字面匹配；
- 对语义改写不敏感；
- 多个 chunk 命中相同关键词时，区分能力有限。

#### `EmbeddingVectorizer`

使用 sentence-transformers 模型生成 dense embedding。

当前默认模型示例：

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

优点：

- 支持语义相似度；
- 更适合中文自然语言 query；
- 更接近真实 RAG 系统中的 dense retrieval。

局限：

- 编码成本更高；
- 小数据集上可能引入弱相关结果；
- embedding similarity 不等价于最终回答相关性。

---

### 4.5 `retrievers/`

V4.0 将检索后端抽象为统一接口：

```python
class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        pass
```

当前支持两种实现。

#### `NumpyRetriever`

使用 numpy 逐个计算 query embedding 与 chunk embedding 的 cosine similarity，然后排序取 top-k。

适合：

- 小数据集；
- 教学原型；
- 结果可解释；
- 与 FAISS 做结果对齐 baseline。

#### `FaissRetriever`

使用 FAISS 构建向量索引。

由于项目原本使用 cosine similarity，因此 FAISS 后端采用：

```text
1. 对 chunk embeddings 做 L2 normalize
2. 使用 faiss.IndexFlatIP
3. 对 query embedding 做同样 normalize
4. inner product 与 cosine similarity 对齐
```

这样可以保证 FAISS 返回结果与原 numpy cosine similarity 的语义一致。

当前只使用最简单的 `IndexFlatIP`，没有引入 IVF、HNSW、PQ、GPU FAISS 或索引持久化。

---

### 4.6 `reranker/`

负责第二阶段重排序。

#### `NoOpReranker`

不进行实际 rerank，只截取前 `top_n` 个候选结果。

作用：

- 作为 baseline；
- 用于对比 rerank 前后的结果；
- 保证 pipeline 在不接模型时也可以稳定运行。

#### `CrossEncoderReranker`

使用 CrossEncoder / BGE reranker 对候选 chunk 重新打分。

流程：

```text
第一阶段 retriever 召回 retrieve_top_k 个 candidates
  ↓
构造 (query, chunk.text) pair
  ↓
CrossEncoder 输出 query-chunk 相关性分数
  ↓
写入 rerank_score
  ↓
按 rerank_score 重新排序
  ↓
返回 rerank_top_n 个最终 chunks
```

rerank 的作用是改善第一阶段检索结果的排序质量，但它不是无条件有效的，并且会带来明显延迟开销。

---

### 4.7 `evaluate.py`

负责检索效果评估。

当前支持：

- `Top1 Hit Rate`
- `Recall@K`
- `MRR`
- `Avg Latency`

同时支持两种标注方式：

```json
{
  "query": "两个向量相不相近一般看什么指标",
  "expected_chunk_ids": ["b_chunk_0"]
}
```

或：

```json
{
  "query": "两个向量相不相近一般看什么指标",
  "expected_keywords": ["余弦相似度", "接近程度"]
}
```

在对比不同 chunking 策略时，`chunk_id` 可能因为切分方式变化而不稳定，因此 V4.0 更推荐使用 `expected_keywords` 做评估。

---

## 5. 项目演进过程

### V1：最小检索闭环

实现了：

- 文档读取；
- fixed chunk 切分；
- keyword vectorizer；
- cosine similarity；
- top-k 检索；
- CLI 运行；
- 基础测试。

目标是先打通最小检索链路。

---

### V2：接入真实 embedding

新增 `EmbeddingVectorizer`，支持 sentence-transformers 模型。

目标是从关键词匹配扩展到语义检索。

---

### V2.1：chunk embedding 预计算

将原来的“每次 query 重新编码所有 chunk”改为：

```text
文档加载后预计算 chunk embeddings
查询时只编码 query
再与已有 chunk embeddings 检索
```

这更接近真实检索系统。

---

### V3：最小 RAG 闭环

新增：

- prompt builder；
- mock generator；
- llm generator 预留接口；
- answer + references 输出。

目标是打通 retrieval → generation 主链路。

---

### V3.2：二阶段 rerank

新增：

- `RetrievedChunk`；
- `BaseReranker`；
- `NoOpReranker`；
- `CrossEncoderReranker`；
- `retrieval_score / rerank_score` 输出。

目标是解决第一阶段检索“能召回但排序不一定最优”的问题。

---

### V3.3：evaluation 与 error analysis

新增：

- `evaluate.py`；
- `eval/questions.json`；
- `Top1 Hit / Recall@K / MRR / Latency`；
- `eval/error_cases.md`。

目标是用指标和失败案例分析系统，而不是只看单条 query 的主观结果。

---

### V4.0：chunking 策略 + FAISS 检索后端

新增：

- `--splitter fixed / paragraph / sentence`；
- `expected_keywords` 评估方式；
- `app/retrievers/` 检索后端抽象；
- `--retriever numpy / faiss`；
- `FaissRetriever`；
- chunking 对比实验；
- numpy / FAISS 结果对齐验证。

目标是进一步增强项目深度：从“能检索”升级为“能分析 chunking 对检索质量的影响，并支持更接近真实系统的向量索引后端”。

---

## 6. 安装依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 至少包含：

```text
numpy
pytest
sentence-transformers
openai
faiss-cpu
```

说明：

- `sentence-transformers` 用于 embedding vectorizer 和 CrossEncoder reranker；
- `openai` 仅在使用 `--generator llm` 时需要；
- `faiss-cpu` 用于 FAISS 检索后端；
- 若只使用 `mock` generator，不需要配置 API key。

如果 Hugging Face 下载模型时出现：

```text
Warning: You are sending unauthenticated requests to the HF Hub.
```

这是正常提示，不影响基本运行。频繁下载模型时可以配置 `HF_TOKEN`。

---

## 7. 运行方式

以下命令均为 Windows PowerShell 单行写法。

### 7.1 fixed splitter + keyword retrieval

```powershell
python main.py --mode retrieve --input_dir data --query "两个向量相不相近一般看什么指标" --splitter fixed --chunk_size 30 --overlap 5 --vectorizer keyword --retriever numpy --retrieve_top_k 5 --reranker none --rerank_top_n 3
```

---

### 7.2 paragraph splitter + keyword retrieval

```powershell
python main.py --mode retrieve --input_dir data --query "两个向量相不相近一般看什么指标" --splitter paragraph --chunk_size 80 --overlap 0 --vectorizer keyword --retriever numpy --retrieve_top_k 5 --reranker none --rerank_top_n 3
```

---

### 7.3 sentence splitter + keyword retrieval

```powershell
python main.py --mode retrieve --input_dir data --query "两个向量相不相近一般看什么指标" --splitter sentence --chunk_size 80 --overlap 1 --vectorizer keyword --retriever numpy --retrieve_top_k 5 --reranker none --rerank_top_n 3
```

---

### 7.4 embedding + numpy retriever

```powershell
python main.py --mode retrieve --input_dir data --query "两个向量相不相近一般看什么指标" --splitter paragraph --chunk_size 80 --overlap 0 --vectorizer embedding --retriever numpy --retrieve_top_k 5 --reranker none --rerank_top_n 3
```

---

### 7.5 embedding + FAISS retriever

```powershell
python main.py --mode retrieve --input_dir data --query "两个向量相不相近一般看什么指标" --splitter paragraph --chunk_size 80 --overlap 0 --vectorizer embedding --retriever faiss --retrieve_top_k 5 --reranker none --rerank_top_n 3
```

---

### 7.6 keyword + CrossEncoder rerank

```powershell
python main.py --mode retrieve --input_dir data --query "两个向量相不相近一般看什么指标" --splitter fixed --chunk_size 30 --overlap 5 --vectorizer keyword --retriever numpy --retrieve_top_k 5 --reranker cross_encoder --reranker_model BAAI/bge-reranker-base --rerank_top_n 3
```

---

### 7.7 RAG mock generator

```powershell
python main.py --mode rag --generator mock --input_dir data --query "两个向量相不相近一般看什么指标" --splitter paragraph --chunk_size 80 --overlap 0 --vectorizer embedding --retriever faiss --retrieve_top_k 5 --reranker none --rerank_top_n 3
```

---

### 7.8 检索评估

```powershell
python evaluate.py --input_dir data --eval_file eval/questions.json --splitter paragraph --chunk_size 80 --overlap 0 --vectorizer embedding --retriever faiss --retrieve_top_k 5 --reranker none --rerank_top_n 3
```

---

## 8. 测试

运行：

```bash
pytest
```

当前测试覆盖：

- splitter 基本行为；
- fixed / paragraph / sentence 切分；
- retriever 返回 `RetrievedChunk`；
- NoOpReranker 行为；
- evaluation 指标函数；
- prompt builder；
- mock generator；
- FAISS retriever 基本行为。

FAISS 测试建议使用：

```python
pytest.importorskip("faiss")
```

这样当环境未安装 FAISS 时，相关测试会被 skip，而不是直接失败。

---

## 9. 实验结果

### 9.1 Chunking 策略对比实验

实验设置：

- vectorizer：`keyword`
- retriever：`numpy`
- reranker：`none`
- retrieve_top_k：5
- rerank_top_n：3
- eval queries：5 条
- 评估方式：`expected_keywords`

实验结果：

| splitter | chunks | Top1 Hit | Recall@3 | MRR | Avg Latency |
|---|---:|---:|---:|---:|---:|
| fixed | 5 | 0.4000 | 0.6000 | 0.5000 | 0.03ms |
| paragraph | 3 | 1.0000 | 1.0000 | 1.0000 | 0.03ms |
| sentence | 3 | 1.0000 | 1.0000 | 1.0000 | 0.02ms |

结果分析：

在当前样例数据中，`fixed` splitter 明显弱于 `paragraph` 和 `sentence`。主要原因是固定长度切分可能破坏句子边界，使完整答案依据被截断。

例如，fixed 切分下，相关 chunk 被截断为：

```text
余弦相似度常用于衡量两个向量方向的接近程度。在检索任务中可以
```

它缺少后半句：

```text
比较 query 和文档。
```

因此对于 query：

```text
怎么判断检索结果和问题是否相关
```

该 chunk 虽然语义相关，但不再包含完整答案依据，导致基于 `expected_keywords = ["query", "文档"]` 的评估失败。

而 paragraph / sentence splitter 能保留完整语义单元：

```text
余弦相似度常用于衡量两个向量方向的接近程度。在检索任务中可以比较 query 和文档。
```

所以指标明显提升。

结论：**chunking 不是简单预处理细节，而是 RAG 检索质量的重要影响因素。**

---

### 9.2 Rerank 实验

在 fixed splitter + keyword retrieval 下，曾出现典型排序问题：

```text
Query: 两个向量相不相近一般看什么指标
```

keyword retrieval 结果：

| rank | chunk_id | retrieval_score | 内容摘要 |
|---:|---|---:|---|
| 1 | a_chunk_1 | 0.5774 | 数值计算。它支持数组和向量操作。 |
| 2 | b_chunk_0 | 0.5774 | 余弦相似度常用于衡量两个向量方向的接近程度。 |

由于两个 chunk 都命中了“向量”，keyword baseline 给出了相同的 retrieval_score，导致弱相关的 `a_chunk_1` 排在前面。

接入 CrossEncoder reranker 后：

| rank | chunk_id | retrieval_score | rerank_score | 内容摘要 |
|---:|---|---:|---:|---|
| 1 | b_chunk_0 | 0.5774 | 0.9133 | 余弦相似度常用于衡量两个向量方向的接近程度。 |
| 2 | a_chunk_1 | 0.5774 | 0.0322 | 数值计算。它支持数组和向量操作。 |

结论：reranker 可以在第一阶段同分或弱区分的候选中，进一步根据 query 与 chunk 的整体语义匹配程度重新排序。

但 rerank 并非总是提升效果。在较小数据集和较短 chunk 场景下，reranker 也可能引入新的错误排序，并且会带来明显推理延迟。因此本项目没有把 rerank 当作默认必开能力，而是通过 evaluation 和 error analysis 判断其收益。

---

### 9.3 FAISS 检索后端验证

在 paragraph splitter + embedding retrieval 下，对比 numpy 和 FAISS：

```text
Query: 两个向量相不相近一般看什么指标
```

numpy retriever：

| rank | chunk_id | retrieval_score |
|---:|---|---:|
| 1 | b_chunk_0 | 0.5021 |
| 2 | a_chunk_0 | 0.1169 |
| 3 | c_chunk_0 | 0.0512 |

FAISS retriever：

| rank | chunk_id | retrieval_score |
|---:|---|---:|
| 1 | b_chunk_0 | 0.5021 |
| 2 | a_chunk_0 | 0.1169 |
| 3 | c_chunk_0 | 0.0512 |

结论：FAISS 后端与 numpy cosine similarity 的排序和分数基本一致，说明“向量归一化 + `IndexFlatIP`”实现正确。

由于当前数据规模只有 3 个 chunks，FAISS 的性能优势不会明显体现。此处接入 FAISS 的主要价值是：

- 抽象 retriever 后端；
- 验证 numpy → FAISS 的工程扩展路径；
- 为后续更大规模 chunk 检索打基础。

---

## 10. 当前局限

当前版本仍然是轻量原型，主要限制包括：

1. 数据集很小，目前只有 3～5 个 chunks；
2. eval queries 数量仍较少，目前只有 5 条；
3. 仅支持 `.txt` 文件；
4. paragraph / sentence splitter 仍然是简化实现，未基于 tokenizer；
5. FAISS 使用的是最简单的 `IndexFlatIP`，没有持久化索引；
6. 每次运行仍会重新构建 chunk embeddings 和索引；
7. `mock` generator 不是真实 LLM 回答；
8. CrossEncoder reranker 延迟较高；
9. 当前主要评估 retrieval quality，还没有系统评估 answer quality；
10. 尚未接入 Qdrant / Milvus 这类服务化向量数据库。

---

## 11. 后续优化方向

### P0：扩大 evaluation set

继续扩充 `eval/questions.json`，覆盖：

- 强关键词 query；
- 语义改写 query；
- 干扰型 query；
- 多答案 query；
- 检索失败 query。

目标是让实验结论更稳定，而不是依赖少量样例。

---

### P0：补充 error analysis

继续记录：

- query；
- expected keywords；
- baseline 排序；
- rerank 排序；
- 是否改善；
- 是否引入错误；
- 失败原因分析。

---

### P1：扩大文档规模

当前数据太小，很多工程能力体现不明显。后续可以加入：

- 操作系统笔记；
- 数据库笔记；
- 计算机网络笔记；
- Python / NumPy 文档片段；
- 项目 README / 技术文档。

扩大文档规模后，再重新比较：

- splitter 策略；
- numpy vs FAISS；
- embedding only vs embedding + rerank。

---

### P1：FAISS 持久化与索引复用

当前每次运行都会重新构建 FAISS index。后续可以考虑：

- 保存 FAISS index；
- 保存 chunk metadata；
- 运行时直接加载索引；
- 支持重新构建索引。

这会让项目更接近真实检索系统。

---

### P2：Qdrant 向量数据库

如果后续希望展示服务化 vector store，可以考虑接入 Qdrant，重点验证：

- 向量持久化；
- metadata filtering；
- collection 管理；
- API 化检索。

当前不急于接 Milvus，因为项目规模还不足以支撑分布式向量数据库的必要性。

---

### P2：真实 LLM 与回答质量评估

后续可以完善：

- 从环境变量读取 API key；
- 真实 LLM 回答生成；
- answer faithfulness 检查；
- 引用与回答一致性检查；
- 回答质量评估。

---

## 12. 面试讲法

### 项目概述

> 这是一个轻量级 RAG 文档检索原型。我没有直接使用 LangChain，而是从底层实现了文档读取、chunk 切分、metadata 管理、keyword / embedding 检索、chunk embedding 预计算、numpy / FAISS 检索后端、二阶段 rerank、prompt 构造、mock generator 和检索评估。项目重点不是堆功能，而是把 RAG 主链路做成可运行、可解释、可评估、可扩展的工程系统。

### 为什么不用 LangChain？

> 这个项目的目标是理解 RAG 主链路中每个模块的作用和取舍。自己实现 splitter、vectorizer、retriever、reranker、prompt builder 和 evaluator 后，可以更清楚地解释 chunking、embedding、top-k retrieval、rerank、上下文构造和评估指标。如果后续接入 LangChain，也能知道它封装了哪些环节，而不是只会调用接口。

### 为什么要优化 chunking？

> chunking 会直接影响检索质量。固定字符长度切分虽然简单，但容易切断句子和语义边界。在实验中，fixed splitter 把“在检索任务中可以比较 query 和文档”截断，导致相关 query 无法命中完整答案依据。paragraph 和 sentence splitter 保留了完整语义单元，因此在当前评估集上 Top1 Hit、Recall@3 和 MRR 都明显更好。

### 为什么要接 FAISS？

> 原始版本使用 numpy 遍历所有 chunk embedding 并逐个计算 cosine similarity，这适合小数据集和教学原型。接入 FAISS 后，检索后端被抽象成统一接口，可以从暴力检索扩展到向量索引检索。我使用向量归一化 + IndexFlatIP，使 FAISS 的 inner product 检索结果与 cosine similarity 对齐。在当前小数据集下性能优势不明显，但它为后续更大规模检索打下了工程基础。

### 为什么 rerank 不一定总是开？

> reranker 能改善部分排序错误，尤其是在第一阶段召回结果同分或弱区分时。但 CrossEncoder 推理成本更高，而且在小数据集、短 chunk 场景下也可能引入错误排序。所以我没有把 rerank 当作默认必开能力，而是通过 evaluation 和 error analysis 分析它在不同 query 上的收益和代价。

---

## 13. 项目总结

V4.0 当前完成了：

- 文档读取；
- fixed / paragraph / sentence 三种 chunking 策略；
- metadata 管理；
- keyword / embedding 双模式向量化；
- chunk embedding 预计算；
- numpy / FAISS 两种检索后端；
- cosine similarity 与 FAISS IndexFlatIP 对齐；
- top-k candidate retrieval；
- `RetrievedChunk` 统一结果结构；
- `NoOpReranker` baseline；
- `CrossEncoderReranker` 二阶段重排序；
- retrieval / rag 双模式；
- prompt 构造；
- mock / llm 双 generator 结构；
- `evaluate.py` 检索评估脚本；
- `expected_chunk_ids / expected_keywords` 双评估方式；
- Top1 Hit、Recall@K、MRR、Latency 指标；
- chunking 策略对比实验；
- rerank 成功与失败案例分析；
- FAISS 检索后端验证；
- 基础测试。

这个版本已经从“最小 RAG demo”升级为一个具备检索实验、模块抽象和工程取舍分析的轻量 RAG 原型。下一步如果继续加深，优先方向不是继续堆模型，而是扩大数据规模、增强 evaluation set、做 FAISS index 持久化，或者进一步接入 Qdrant 做服务化向量存储。
