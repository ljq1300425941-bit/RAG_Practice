# 简易文档检索 / RAG 原型（README V3.3）

一个基于 Python 实现的轻量级文档检索 / RAG 原型。项目从底层实现了文档读取、chunk 切分、向量化、top-k 召回、二阶段 rerank、prompt 构造、mock / llm generator 以及检索评估流程。

当前版本重点不是堆叠 LangChain、向量数据库或复杂 Agent，而是把 RAG 主链路做成一个可运行、可解释、可评估、可扩展的小型工程项目。

```text
文档读取
  → chunk 切分
  → metadata 管理
  → 向量化
  → chunk embedding 预计算
  → 第一阶段 top-k 召回
  → 第二阶段 rerank 重排序
  → prompt 构造
  → mock / llm generator
  → answer + references 输出
  → evaluation + error analysis
```

---

## 1. 项目功能

当前版本支持：

- 读取指定目录下的多个 `.txt` 文件；
- 将文本按固定长度切分为多个 chunk；
- 为每个 chunk 保存基础 metadata；
- 支持两种向量化方式：
  - `keyword`：关键词计数向量；
  - `embedding`：真实 embedding 向量；
- 对 chunk embeddings 做预计算，避免查询时重复编码全部文档；
- 使用余弦相似度计算 query 与 chunk 的相关性；
- 支持第一阶段 top-k candidate retrieval；
- 支持第二阶段 rerank：
  - `none`：不重排，作为 baseline；
  - `cross_encoder`：使用 CrossEncoder / BGE reranker 对候选 chunk 重新打分；
- 支持两种运行模式：
  - `retrieve`：仅输出检索 / rerank 结果；
  - `rag`：检索后构造上下文并生成回答；
- 支持两种生成方式：
  - `mock`：基于检索结果输出回答草稿；
  - `llm`：预留真实 LLM generator 接口；
- 输出 `retrieval_score` 和 `rerank_score`，方便分析 rerank 前后的排序变化；
- 支持检索评估：
  - `Top1 Hit Rate`
  - `Recall@K`
  - `MRR`
  - `Avg Latency`
- 记录 rerank 成功案例和失败案例，便于分析工程取舍。

---

## 2. 项目结构

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
    test_prompt_builder.py
    test_generator.py
  requirements.txt
  README.md
```

其中：

- `app/`：核心代码模块；
- `app/reranker/`：二阶段重排序模块；
- `evaluate.py`：检索效果评估脚本；
- `eval/questions.json`：人工标注的评估 query；
- `eval/error_cases.md`：成功案例与失败案例分析。

---

## 3. 核心模块说明

### 3.1 `models.py`

定义文档和向量相关的基础数据结构：

- `DocumentChunk`
- `ChunkEmbedding`

`DocumentChunk` 表示切分后的文本块，包含：

- `chunk_id`
- `source_file`
- `text`
- `start_pos`
- `end_pos`

`ChunkEmbedding` 表示一个 chunk 及其对应的向量表示。

---

### 3.2 `schema.py`

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

这样可以同时保留：

- 第一阶段召回分数 `retrieval_score`；
- 第二阶段重排分数 `rerank_score`；
- chunk 的文本和来源信息。

这也让后续扩展 FAISS、Qdrant、BM25、hybrid search 时接口更稳定。

---

### 3.3 `splitter.py`

负责文本切分与 chunk 构建：

- 读取 `.txt` 文本；
- 按 `chunk_size` 和 `overlap` 切分；
- 构建带 metadata 的 `DocumentChunk`。

当前采用固定长度切分，优点是简单、可控；局限是无法感知句子、段落或语义边界。

---

### 3.4 `vectorizer.py`

负责文本转向量。

#### `KeywordCountVectorizer`

一个简化的 bag-of-words baseline。它统计词表中关键词在文本中出现的次数，并组成固定长度向量。

优点：

- 实现简单；
- 可解释性强；
- 在强关键词场景下效果稳定；
- 适合作为 baseline。

局限：

- 依赖字面词匹配；
- 对语义改写不敏感；
- 当多个 chunk 命中相同关键词时，排序区分能力较弱。

#### `EmbeddingVectorizer`

使用 sentence-transformers 模型生成 dense embedding。

当前主要使用：

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

优点：

- 更适合语义匹配；
- 能处理不完全同词的 query 和 chunk；
- 更接近真实 RAG 系统中的 dense retrieval。

局限：

- 编码成本更高；
- 小数据集上可能引入弱相关召回；
- embedding 相似度不等价于最终回答相关性。

---

### 3.5 `retriever.py`

负责第一阶段检索：

1. 对 query 编码；
2. 使用预计算好的 chunk embeddings；
3. 计算余弦相似度；
4. 返回 `retrieve_top_k` 个候选 `RetrievedChunk`。

当前流程已经从“每次查询重复编码所有 chunk”升级为：

```text
文档加载后先编码全部 chunks
查询时只编码 query
query embedding 与 chunk embeddings 计算相似度
返回 retrieve_top_k 个候选结果
```

---

### 3.6 `reranker/`

负责第二阶段重排序。

#### `BaseReranker`

定义统一接口：

```python
class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int
    ) -> list[RetrievedChunk]:
        pass
```

#### `NoOpReranker`

不进行实际重排，只截取前 `top_n` 个结果，作为 baseline。

```text
retriever 原始排序
  → NoOpReranker
  → 原样返回前 top_n
```

#### `CrossEncoderReranker`

使用 CrossEncoder / BGE reranker 对候选 chunk 重新打分。

当前主要使用：

```text
BAAI/bge-reranker-base
```

基本流程：

```text
第一阶段 retriever 召回 retrieve_top_k 个 candidates
  ↓
构造 (query, chunk.text) pair
  ↓
CrossEncoder 判断 query 与 chunk 的相关性
  ↓
写入 rerank_score
  ↓
按 rerank_score 排序
  ↓
返回 rerank_top_n 个最终 chunks
```

这种方式通常比直接用 embedding similarity 判断相关性更细致，但计算成本也更高，因此只适合对少量候选结果 rerank。

---

### 3.7 `prompt_builder.py`

负责将 query 和最终 chunks 构造成 RAG prompt。

当前 prompt 要求：

- 只能根据给定上下文回答；
- 信息不足时说明无法确定；
- 不编造上下文中没有的信息；
- 保留 chunk 来源，方便输出 references。

---

### 3.8 `generator.py`

负责回答生成。

当前支持两种 generator：

#### `mock`

不调用真实 LLM，直接基于检索结果组织回答草稿。

价值：

- 不依赖外部 API；
- 方便本地稳定演示；
- 可以观察 retrieval / rerank 结果如何影响最终上下文；
- 适合项目主链路验证。

#### `llm`

预留真实 LLM 接口。当前版本结构上支持接入真实模型，但项目主演示路径仍以 `mock` 为主，避免把重点转移到 API 配额、账单和平台配置上。

---

## 4. 核心流程

当前版本整体流程如下：

```text
输入文档目录
   ↓
读取多个 .txt 文件
   ↓
文本切分为 chunks
   ↓
构建 DocumentChunk metadata
   ↓
选择向量化方式（keyword / embedding）
   ↓
预计算 chunk embeddings
   ↓
query 编码
   ↓
第一阶段检索：余弦相似度排序，得到 retrieve_top_k 个 candidates
   ↓
第二阶段 rerank：对 query-chunk pair 重新打分
   ↓
得到 rerank_top_n 个最终 chunks
   ↓
若 mode=retrieve：直接输出检索与 rerank 结果
   ↓
若 mode=rag：构造 prompt
   ↓
根据 generator 生成回答
   ↓
输出 answer 与 references
```

---

## 5. 项目演进过程

### V1：最小检索闭环

实现文档读取、chunk 切分、metadata 管理、关键词计数向量、余弦相似度、top-k 检索和基础测试。

目标是先把检索链路跑通。

---

### V2：接入真实 embedding

在 keyword baseline 基础上，接入 sentence-transformers embedding 模型，增强语义检索能力。

目标是对比 keyword retrieval 和 embedding retrieval 在不同 query 下的表现。

---

### V2.1：chunk embedding 预计算

将 chunk embedding 的计算从查询阶段前移到文档加载后，查询时只编码 query。

目标是减少重复计算，让流程更接近真实检索系统。

---

### V3：最小 RAG 闭环

补全 prompt builder 和 generator，使项目从 retrieval demo 变成最小 RAG 闭环。

---

### V3.1：保留 mock generator，预留真实 LLM 接口

将 `mock` 作为主演示路径，将 `llm` 作为预留能力，保证项目可运行性和演示稳定性。

---

### V3.2：引入二阶段检索与 rerank

新增：

- `RetrievedChunk`
- `BaseReranker`
- `NoOpReranker`
- `CrossEncoderReranker`
- `--retrieve_top_k`
- `--reranker`
- `--reranker_model`
- `--rerank_top_n`

目标是解决第一阶段召回结果“能召回但排序不一定最优”的问题。

---

### V3.3：加入 evaluation 与 error analysis

新增：

- `evaluate.py`
- `eval/questions.json`
- `eval/error_cases.md`
- Top1 Hit、Recall@K、MRR、Latency 指标；
- keyword / embedding / rerank 四组实验对比；
- rerank 成功案例与失败案例分析。

这一阶段的重点是从“我加了 rerank”升级为：

> 我能量化比较不同检索策略，也能分析 rerank 的收益、失败和延迟代价。

---

## 6. 环境依赖

安装依赖：

```bash
pip install -r requirements.txt
```

当前依赖至少包括：

```text
numpy
pytest
sentence-transformers
openai
```

说明：

- `sentence-transformers` 用于 embedding vectorizer 和 CrossEncoder reranker；
- `openai` 仅在使用 `--generator llm` 时需要；
- 默认演示路径是 `mock` generator，因此不配置 API key 也可以跑通主流程。

如果使用 Hugging Face 模型时出现：

```text
Warning: You are sending unauthenticated requests to the HF Hub.
```

这是未登录 Hugging Face 的提示，不影响基本使用。后续频繁下载模型时可以配置 `HF_TOKEN`。

---

## 7. 运行方式

以下命令均为 Windows PowerShell 单行写法。

### 7.1 keyword baseline，不使用 rerank

```powershell
python main.py --mode retrieve --input_dir data --query "向量接近程度怎么衡量" --chunk_size 30 --overlap 5 --vectorizer keyword --retrieve_top_k 10 --reranker none --rerank_top_n 3
```

### 7.2 keyword + CrossEncoder rerank

```powershell
python main.py --mode retrieve --input_dir data --query "向量接近程度怎么衡量" --chunk_size 30 --overlap 5 --vectorizer keyword --retrieve_top_k 5 --reranker cross_encoder --reranker_model BAAI/bge-reranker-base --rerank_top_n 3
```

### 7.3 embedding retrieval，不使用 rerank

```powershell
python main.py --mode retrieve --input_dir data --query "向量接近程度怎么衡量" --chunk_size 30 --overlap 5 --vectorizer embedding --model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --retrieve_top_k 10 --reranker none --rerank_top_n 3
```

### 7.4 embedding + CrossEncoder rerank

```powershell
python main.py --mode retrieve --input_dir data --query "向量接近程度怎么衡量" --chunk_size 30 --overlap 5 --vectorizer embedding --model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --retrieve_top_k 5 --reranker cross_encoder --reranker_model BAAI/bge-reranker-base --rerank_top_n 3
```

### 7.5 RAG 模式：mock generator

```powershell
python main.py --mode rag --generator mock --input_dir data --query "向量接近程度怎么衡量" --chunk_size 30 --overlap 5 --vectorizer keyword --retrieve_top_k 5 --reranker cross_encoder --reranker_model BAAI/bge-reranker-base --rerank_top_n 3
```

### 7.6 RAG 模式：llm generator

```powershell
python main.py --mode rag --generator llm --input_dir data --query "向量接近程度怎么衡量" --chunk_size 30 --overlap 5 --vectorizer embedding --model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --retrieve_top_k 10 --reranker none --rerank_top_n 3 --llm_model gpt-4.1-mini
```

说明：

- `llm` generator 需要 API key；
- 项目主演示路径建议使用 `mock`；
- 若继续推进真实生成，建议通过环境变量配置 API key，而不是在命令行明文传入。

---

## 8. Evaluation

项目新增 `evaluate.py`，用于评估检索与 rerank 效果。

### 8.1 评估指标

当前使用四个指标：

| 指标 | 含义 |
|---|---|
| Top1 Hit Rate | 正确 chunk 是否排在第一位 |
| Recall@3 | 正确 chunk 是否出现在前 3 个结果中 |
| MRR | 正确 chunk 排名越靠前，分数越高 |
| Avg Latency | 平均查询耗时 |

### 8.2 评估命令

#### keyword only

```powershell
python evaluate.py --input_dir data --eval_file eval/questions.json --chunk_size 30 --overlap 5 --vectorizer keyword --retrieve_top_k 5 --reranker none --rerank_top_n 3
```

#### keyword + rerank

```powershell
python evaluate.py --input_dir data --eval_file eval/questions.json --chunk_size 30 --overlap 5 --vectorizer keyword --retrieve_top_k 5 --reranker cross_encoder --rerank_top_n 3
```

#### embedding only

```powershell
python evaluate.py --input_dir data --eval_file eval/questions.json --chunk_size 30 --overlap 5 --vectorizer embedding --retrieve_top_k 5 --reranker none --rerank_top_n 3
```

#### embedding + rerank

```powershell
python evaluate.py --input_dir data --eval_file eval/questions.json --chunk_size 30 --overlap 5 --vectorizer embedding --retrieve_top_k 5 --reranker cross_encoder --rerank_top_n 3
```

---

## 9. 实验结果

### 9.1 实验设置

- 数据规模：5 个 chunks；
- 评估 query 数量：5 条；
- `retrieve_top_k=5`；
- `rerank_top_n=3`；
- embedding model：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`；
- reranker model：`BAAI/bge-reranker-base`。

### 9.2 Summary

| 方法 | Top1 Hit | Recall@3 | MRR | Avg Latency |
|---|---:|---:|---:|---:|
| keyword only | 0.8000 | 1.0000 | 0.9000 | 0.03ms |
| keyword + rerank | 0.8000 | 1.0000 | 0.9000 | 154.12ms |
| embedding only | 0.8000 | 1.0000 | 0.9000 | 28.76ms |
| embedding + rerank | 0.8000 | 1.0000 | 0.9000 | 163.30ms |

### 9.3 结果分析

从当前小评估集可以得到以下结论：

1. 四种方法的 `Recall@3` 都达到 `1.0000`，说明第一阶段召回基本能够把正确 chunk 放入候选集合中。
2. `Top1 Hit` 和 `MRR` 在四种方法中均为 `0.8000` 和 `0.9000`，说明 rerank 在当前小数据集上没有带来整体指标提升。
3. rerank 修正了一个 keyword 排序错误：对于 query `两个向量相不相近一般看什么指标`，keyword 检索将弱相关的 `a_chunk_1` 排在第一，而 reranker 将真正相关的 `b_chunk_0` 提升到了第一。
4. rerank 也引入了一个新的排序错误：对于 query `怎么判断检索结果和问题是否相关`，reranker 将不相关的 `c_chunk_0` 提升到了第一。
5. rerank 带来了明显延迟开销。keyword only 的平均延迟约为 `0.03ms`，keyword + rerank 的平均延迟约为 `154.12ms`。

因此当前结论不是“rerank 显著提升了整体准确率”，而是：

> rerank 可以改善部分候选排序问题，但在当前小数据集上收益不稳定，并且会带来明显延迟开销。

---

## 10. Error Analysis

详细错误案例见：

```text
eval/error_cases.md
```

当前记录了两个典型案例：

1. keyword 同分导致排序错误，rerank 成功修正；
2. rerank 将不相关 chunk 错误提升到第一，导致 Top1 变差。

这部分的意义是说明项目不是只展示成功样例，而是分析 rerank 的边界和风险。

---

## 11. 测试

在项目根目录运行：

```bash
pytest
```

当前测试已覆盖：

- `splitter`
- `retriever`
- `prompt_builder`
- `generator`

后续建议补充：

- `NoOpReranker` 测试；
- `CrossEncoderReranker` mock 测试；
- evaluation metrics 测试。

---

## 12. 当前版本局限

当前版本仍然是一个轻量 RAG 原型，主要限制包括：

1. 数据集很小，目前只有 5 个 chunks；
2. 评估 query 数量较少，目前只有 5 条；
3. 仅支持 `.txt` 文件；
4. 仅支持固定长度 chunking；
5. 没有持久化索引，每次运行仍会重新构建 chunk embeddings；
6. 没有接入 FAISS、Qdrant、Milvus 等向量索引或向量数据库；
7. `mock` generator 不是真实 LLM 回答；
8. CrossEncoder reranker 延迟较高，不适合对全量文档直接重排；
9. 当前只评估 retrieval quality，还没有系统评估 answer quality。

---

## 13. 后续优化方向

后续优先级建议如下：

### P0：扩大 evaluation set

继续增加 query 数量，覆盖：

- 强关键词 query；
- 语义改写 query；
- 干扰性 query；
- 多答案 query；
- 检索失败 query。

目标是让评估结果更稳定，而不是依赖少量样例。

---

### P0：补充 error analysis

继续记录：

- query；
- expected chunks；
- baseline 排序；
- rerank 排序；
- 是否改善；
- 是否引入错误；
- 失败原因分析。

---

### P1：优化 chunking

当前固定长度切分可能切断语义边界。后续可以尝试：

- 按句子切分；
- 按段落切分；
- 基于 token 数切分；
- 更合理的 overlap 策略。

---

### P1：接入 FAISS

当 chunk 数量扩大后，可以用 FAISS 替代 numpy 暴力相似度计算。

当前不急于接入 Qdrant / Milvus，因为项目核心问题仍然是检索质量、排序质量和评估方式。

---

### P2：完善真实 LLM 生成

后续可以继续完善：

- API key 配置；
- 真实回答生成；
- answer quality 评估；
- 引用与回答一致性检查。

---

## 14. 面试讲法

可以这样介绍项目：

> 这是一个轻量级 RAG 文档检索原型。我没有直接使用 LangChain，而是从底层实现了文档读取、chunk 切分、metadata 管理、keyword / embedding 检索、chunk embedding 预计算、二阶段 rerank、prompt 构造和 mock generator。后续我又加入了 `evaluate.py`，用 Top1 Hit、Recall@3、MRR 和平均延迟评估不同检索策略。当前实验表明，rerank 能修正部分 keyword 排序错误，但在小数据集上整体指标没有提升，并且会带来明显延迟，因此需要结合 evaluation 和 error analysis 判断是否值得使用。

如果被问“为什么不用 LangChain”，可以回答：

> 这个项目目标不是快速搭一个黑盒应用，而是理解 RAG 主链路中每个模块的作用和取舍。自己实现 splitter、retriever、reranker、prompt builder 和 evaluator 后，更容易解释 chunking、embedding、top-k retrieval、rerank、上下文构造和评估指标。后续如果接入 LangChain，也能清楚知道它封装了哪些环节。

如果被问“为什么不直接接向量数据库”，可以回答：

> 当前数据规模很小，瓶颈不是向量存储能力，而是检索质量、排序质量和评估方式。因此我先用本地 embedding + cosine similarity 打通链路，再加入 rerank 和 evaluation。等数据规模变大或需要 metadata filtering、持久化、服务化检索时，再接 FAISS 或 Qdrant 会更合理。

如果被问“rerank 有什么作用”，可以回答：

> 第一阶段 retriever 负责快速召回候选，但排序不一定最准确。reranker 会把 query 和候选 chunk 一起输入模型，重新判断相关性。比如在 `两个向量相不相近一般看什么指标` 这个例子里，keyword 检索把“支持数组和向量操作”和“余弦相似度衡量向量接近程度”打成同分，但 reranker 能把真正回答问题的 chunk 提升到第一。不过 rerank 并不总是有收益，也可能引入错误排序，所以我用 evaluation 和 error analysis 来分析它的实际效果。

---

## 15. 项目总结

当前项目已经从一个最小检索 demo 升级为一个具备基础工程结构和评估闭环的轻量 RAG 原型，完成了：

- 文档读取；
- 文本切分；
- metadata 管理；
- keyword / embedding 双模式向量化；
- 余弦相似度计算；
- top-k candidate retrieval；
- chunk embedding 预计算；
- `RetrievedChunk` 统一结果结构；
- `NoOpReranker` baseline；
- `CrossEncoderReranker` 二阶段重排序；
- retrieval / rag 双模式；
- prompt 构造；
- mock / llm 双 generator 结构；
- retrieval_score / rerank_score 输出；
- `evaluate.py` 检索评估脚本；
- Top1 Hit、Recall@3、MRR、Latency 指标；
- 成功案例与失败案例分析；
- 基础测试。

下一步最值得做的不是继续堆新组件，而是扩大 evaluation set、补充 error analysis，并在更真实的数据规模上观察 rerank 是否值得使用。
