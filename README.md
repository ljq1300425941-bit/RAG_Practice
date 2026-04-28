# 简易文档检索 / RAG 原型（README V3.2）

一个基于 Python 实现的简易文档检索 / RAG 原型，支持从目录中读取多个 `.txt` 文本文件，进行 chunk 切分、向量化、相似度计算、top-k 检索、二阶段 rerank，并在此基础上构建最小 RAG 闭环。

当前版本重点不在于堆叠 LangChain、向量数据库或复杂 Agent，而是逐步完成一条清晰、可运行、可扩展、可解释的工程链路：

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
```

本项目目前更偏向“轻量级 RAG 检索链路原型”，重点关注 retrieval、rerank、prompt 构造和工程结构，而不是直接依赖成熟框架隐藏实现细节。

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
- 输出 retrieval_score 和 rerank_score，方便分析 rerank 前后的排序变化；
- 提供基础测试，覆盖 splitter、retriever、prompt builder、generator 等主流程。

---

## 2. 项目结构

```text
RAG_Practice/
  main.py
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
  tests/
    test_splitter.py
    test_retriever.py
    test_prompt_builder.py
    test_generator.py
  requirements.txt
  README.md
```

其中，`reranker/` 是 V3.2 新增的核心模块，用于支持二阶段检索中的重排序逻辑。

---

## 3. 核心模块说明

### `models.py`

定义项目中与文档和向量相关的基础数据结构：

- `DocumentChunk`
- `ChunkEmbedding`

其中：

- `DocumentChunk` 表示切分后的文本块，包含：
  - `chunk_id`
  - `source_file`
  - `text`
  - `start_pos`
  - `end_pos`

- `ChunkEmbedding` 表示：
  - 一个 `DocumentChunk`
  - 它对应的向量表示 `embedding`

这样可以避免在代码中维护容易错位的 parallel list。

---

### `schema.py`

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

引入 `RetrievedChunk` 后，retriever 和 reranker 之间不再直接传递 `(chunk, score)` 这类 tuple，而是传递结构化对象。

这样做的好处是：

- retrieval_score 和 rerank_score 可以同时保留；
- prompt builder / generator 可以直接复用 rerank 后的结果；
- 后续扩展 FAISS、Qdrant、BM25、hybrid search 时，接口更稳定；
- 结果输出更适合调试和实验分析。

---

### `loader.py`

负责文件读取：

- 扫描输入目录下的 `.txt` 文件；
- 返回文件路径列表。

---

### `splitter.py`

负责文本切分与 chunk 构建：

- 将原始文本按 `chunk_size` 和 `overlap` 切分；
- 构建 `DocumentChunk`；
- 支持从目录批量构建 chunk 列表。

当前版本采用固定长度切分，优点是实现简单、行为可控；局限是不能很好地感知句子、段落或语义边界。

---

### `vectorizer.py`

负责文本转向量。

当前版本支持两种模式。

#### 1）`KeywordCountVectorizer`

使用关键词计数向量，本质上是一个非常简化的 bag-of-words 方案。

例如给定词表：

```python
["python", "numpy", "向量", "相似度", "数据库", "检索"]
```

文本中每个词出现的次数会组成一个固定长度向量。

优点：

- 实现简单；
- 易于理解；
- 适合作为 baseline；
- 在强关键词命中场景下表现直观。

局限：

- 更依赖字面词匹配；
- 对表达变化和语义相近表达不敏感；
- 当多个 chunk 命中相同关键词时，排序区分能力较弱。

#### 2）`EmbeddingVectorizer`

使用真实 embedding 模型对文本进行编码。

当前实验中引入了多语言 sentence-transformers 模型，用于提升中文场景下的语义检索能力。

优点：

- 对语义表达变化更敏感；
- 能处理不完全同词的查询与文本匹配；
- 更接近真实 RAG 系统中的 dense retrieval。

局限：

- 编码成本更高；
- 小数据集上仍可能引入弱相关噪声结果；
- embedding 相似度并不等价于最终回答相关性，因此仍需要 rerank 或评估来进一步判断结果质量。

---

### `retriever.py`

负责第一阶段检索逻辑：

- 计算余弦相似度；
- 构建 chunk embeddings；
- 对 query 编码；
- 基于预计算好的 chunk embeddings 做 top-k candidate retrieval；
- 返回 `RetrievedChunk` 列表。

当前版本的检索流程已经从“每次查询现场编码全部 chunk”升级为：

```text
文档加载后先编码全部 chunks
查询时只编码 query
再与已有 chunk embeddings 计算相似度
返回 retrieve_top_k 个候选 chunks
```

这样更接近真实检索系统的工作方式。

---

### `reranker/`

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

不进行实际重排，只截取前 `top_n` 个候选结果。

它的作用是作为 baseline：

```text
retriever 原始排序
  → NoOpReranker
  → 原样返回前 top_n
```

这样可以方便对比：

- 未 rerank 的结果；
- 接入 cross encoder reranker 后的结果。

#### `CrossEncoderReranker`

使用 CrossEncoder / BGE reranker 对候选 chunk 重新打分。

基本流程：

```text
第一阶段 retriever 召回 retrieve_top_k 个 candidates
  ↓
构造 (query, chunk.text) pair
  ↓
CrossEncoder 直接判断 query 与 chunk 的相关性
  ↓
写入 rerank_score
  ↓
按 rerank_score 重新排序
  ↓
返回 rerank_top_n 个最终 chunks
```

这种方式比单纯 embedding similarity 更慢，但判断 query-chunk 相关性通常更细致，因此适合只对少量候选结果进行重排序。

---

### `prompt_builder.py`

负责构造 RAG prompt。

输入：

- 用户问题 `query`
- rerank 后的 top-n chunks

输出：

- 一个结构化 prompt，用于后续回答生成。

当前 prompt 约束包括：

- 仅根据给定上下文回答；
- 信息不足时明确说明无法确定；
- 不编造上下文中没有的信息。

---

### `generator.py`

负责回答生成。

当前版本支持两种生成方式。

#### 1）`mock`

不依赖外部 API，直接基于检索结果组织出回答草稿。

优点：

- 便于本地演示；
- 不依赖 API key；
- 可以快速验证 RAG 主链路是否打通；
- 方便观察 retrieval / rerank 结果如何影响最终上下文。

#### 2）`llm`

预留真实 LLM generator 接口，结构上已经支持接入真实模型生成回答。

当前版本由于更关注检索链路、rerank 和工程结构，因此项目默认以 `mock` 方式作为主演示路径，`llm` 作为后续升级方向保留。

---

### `cli.py`

负责命令行参数解析。

当前支持参数：

- `--input_dir`
- `--query`
- `--chunk_size`
- `--overlap`
- `--top_k`
- `--retrieve_top_k`
- `--reranker`
- `--reranker_model`
- `--rerank_top_n`
- `--vectorizer`
- `--model_name`
- `--mode`
- `--generator`
- `--llm_model`
- `--api_key`
- `--base_url`

说明：

- `--retrieve_top_k` 表示第一阶段召回多少个候选 chunk；
- `--rerank_top_n` 表示 rerank 后保留多少个最终 chunk；
- `--top_k` 是早期参数，目前建议逐步用 `--retrieve_top_k + --rerank_top_n` 替代。

---

### `main.py`

项目入口文件，负责串联完整流程：

1. 读取命令行参数；
2. 构建 chunks；
3. 选择 vectorizer；
4. 预计算 chunk embeddings；
5. 执行第一阶段 retrieval，得到 candidates；
6. 根据 `--reranker` 构造 reranker；
7. 对 candidates 执行 rerank；
8. 根据 `mode` 选择：
   - `retrieve`：输出 rerank 后的检索结果；
   - `rag`：构造 prompt 并生成回答；
9. 输出 answer 与 references。

---

## 4. 项目演进过程

### V1：最小检索闭环

第一阶段实现了一个最小可运行的检索原型，包括：

- 文档读取；
- chunk 切分；
- metadata 管理；
- 关键词计数向量；
- 余弦相似度计算；
- top-k 检索；
- 命令行运行；
- 基础测试。

这一阶段的目标不是追求高语义能力，而是先把检索链路跑通。

---

### V2：接入真实 embedding

第二阶段在保留 `keyword` 模式的基础上，新增了 `embedding` 模式。

最初接入英文向模型时，在中文 query 场景下结果不够稳定，因此进一步切换到更适合多语言场景的 embedding 模型，中文检索结果更合理。

这一阶段的重点是：

- 保留 V1 关键词方案，作为 baseline；
- 引入真实 embedding，提升语义检索能力；
- 做 keyword / embedding 对比分析。

---

### V2.1：增加 chunk embedding 预计算

在 embedding 模式下，如果每次 query 都重新编码所有 chunks，会带来明显重复计算。

因此项目进一步加入：

- 文档 chunk embeddings 预计算；
- 查询时仅编码 query；
- 检索阶段直接复用已有 chunk embeddings。

这一升级使得项目流程更接近真实检索系统。

---

### V3：最小 RAG 闭环

在 retrieval 稳定后，第三阶段继续向上补全 generation 链路，加入：

- top-k chunk 检索；
- prompt 构造；
- 基于检索结果生成回答；
- 输出引用来源。

这一阶段的重点不是追求复杂回答能力，而是先把 `retrieval → generation` 的主链路打通。

---

### V3.1：保留 mock generator，预留真实 LLM 接口

在 V3 基础上，项目进一步支持：

- `mock` generator：用于稳定演示最小 RAG 闭环；
- `llm` generator：保留真实 LLM 接口与参数入口。

考虑到当前阶段更关注项目完整度、可运行性和可讲解性，而不希望在 API 配额、账单与平台配置上投入过多时间，因此当前版本将：

- `mock` 作为主演示路径；
- `llm` 作为预留升级能力。

---

### V3.2：引入二阶段检索与 rerank

在 V3.1 的基础上，项目进一步加入了 reranker 抽象和 CrossEncoder reranker。

升级点包括：

- 新增 `RetrievedChunk` 统一结果结构；
- 新增 `BaseReranker` 抽象接口；
- 新增 `NoOpReranker` 作为 baseline；
- 新增 `CrossEncoderReranker` 支持二阶段重排序；
- CLI 增加：
  - `--retrieve_top_k`
  - `--reranker`
  - `--reranker_model`
  - `--rerank_top_n`
- 输出中同时展示：
  - `retrieval_score`
  - `rerank_score`
- 支持对比“只检索”和“检索 + rerank”的排序差异。

这一阶段的重点是解决：

> 第一阶段向量或关键词检索能召回相关候选，但不一定能把最适合回答问题的 chunk 排到最前面。

---

## 5. 核心流程

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

## 6. 环境依赖

安装依赖：

```bash
pip install -r requirements.txt
```

当前 `requirements.txt` 至少可包含：

```text
numpy
pytest
sentence-transformers
openai
```

说明：

- `sentence-transformers` 用于 embedding vectorizer 和 CrossEncoder reranker；
- `openai` 仅在使用 `--generator llm` 时需要；
- 如果当前仅使用 `mock` generator，本地即使暂不配置 API 也可以正常演示核心流程。

如果使用 Hugging Face 模型时出现未登录提示，例如：

```text
Warning: You are sending unauthenticated requests to the HF Hub.
```

这是正常提示，不影响基本使用。后续如果频繁下载模型，可以配置 `HF_TOKEN` 提高下载稳定性和速度。

---

## 7. 运行方式

以下命令均为 Windows PowerShell 单行写法，直接复制即可。

### 7.1 仅检索模式：keyword baseline，不使用 rerank

```powershell
python main.py --mode retrieve --input_dir data --query "向量接近程度怎么衡量" --chunk_size 30 --overlap 5 --vectorizer keyword --retrieve_top_k 10 --reranker none --rerank_top_n 3
```

---

### 7.2 RAG 模式：keyword baseline，不使用 rerank

```powershell
python main.py --mode rag --generator mock --input_dir data --query "向量接近程度怎么衡量" --chunk_size 30 --overlap 5 --vectorizer keyword --retrieve_top_k 10 --reranker none --rerank_top_n 3
```

---

### 7.3 仅检索模式：keyword + CrossEncoder rerank

```powershell
python main.py --mode retrieve --input_dir data --query "向量接近程度怎么衡量" --chunk_size 30 --overlap 5 --vectorizer keyword --retrieve_top_k 5 --reranker cross_encoder --reranker_model BAAI/bge-reranker-base --rerank_top_n 3
```

---

### 7.4 RAG 模式：keyword + CrossEncoder rerank

```powershell
python main.py --mode rag --generator mock --input_dir data --query "向量接近程度怎么衡量" --chunk_size 30 --overlap 5 --vectorizer keyword --retrieve_top_k 5 --reranker cross_encoder --reranker_model BAAI/bge-reranker-base --rerank_top_n 3
```

---

### 7.5 仅检索模式：embedding retrieval

```powershell
python main.py --mode retrieve --input_dir data --query "向量接近程度怎么衡量" --chunk_size 30 --overlap 5 --vectorizer embedding --model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --retrieve_top_k 10 --reranker none --rerank_top_n 3
```

---

### 7.6 RAG 模式：embedding retrieval + mock generator

```powershell
python main.py --mode rag --generator mock --input_dir data --query "向量接近程度怎么衡量" --chunk_size 30 --overlap 5 --vectorizer embedding --model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --retrieve_top_k 10 --reranker none --rerank_top_n 3
```

---

### 7.7 预留的真实生成模式：llm generator

```powershell
python main.py --mode rag --generator llm --input_dir data --query "向量接近程度怎么衡量" --chunk_size 30 --overlap 5 --vectorizer embedding --model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --retrieve_top_k 10 --reranker none --rerank_top_n 3 --llm_model gpt-4.1-mini
```

说明：

- 当前版本结构上支持 `llm` generator；
- 项目主演示路径默认使用 `mock`；
- 若后续继续推进 `llm`，建议使用环境变量配置 API key，而不是在命令行中明文传入。

---

## 8. 测试

在项目根目录运行：

```bash
pytest
```

当前测试已覆盖：

- `splitter`
- `retriever`
- `prompt_builder`
- `generator`

在当前版本中，测试主要验证的是：

- 检索与切分主流程正确；
- prompt 构造结果包含 query / chunk / metadata；
- mock generator 在有无上下文时都能正常返回结果。

后续建议补充：

- `NoOpReranker` 测试；
- `CrossEncoderReranker` 的轻量 mock 测试；
- evaluation metrics 测试。

---

## 9. 效果对比与分析

### 9.1 keyword 与 embedding 的对比

基于 4 组 query，对 `keyword` 与 `embedding` 两种模式做了对比。

#### Query 1：`numpy 向量 相似度`

- `keyword` 与 `embedding` 都能返回较合理的结果；
- 两者的 Top 1 都是与“相似度 / 向量”最相关的 chunk；
- 这一组说明：当 query 的关键词非常明确时，keyword baseline 本身也能工作得不错。

结论：

- 强关键词场景下，两者差距不大；
- embedding 没有显著拉开优势，但也没有破坏结果。

---

#### Query 2：`向量接近程度怎么衡量`

- `keyword` 更依赖字面词匹配，把“支持数组和向量操作”的 chunk 排到了第一；
- `embedding` 能把“接近程度怎么衡量”与“余弦相似度常用于衡量两个向量方向的接近程度”对应起来；
- 这一组最能体现语义检索的优势。

结论：

- 在表达变化较大但语义相关的 query 上，embedding 明显更合理。

---

#### Query 3：`数据库 事务`

- 两种模式都能把数据库相关 chunk 排到第一；
- `keyword` 后续结果更“硬匹配”，很多非相关项分数直接为 0；
- `embedding` 在后续结果中引入了一些弱相关内容。

结论：

- embedding 不一定在所有场景下都更“干净”；
- 在小数据集里，embedding 可能会带来更宽松的召回，也会夹带一定噪声。

---

#### Query 4：`Python 数值计算`

- 两种模式都能正确把 `a_chunk_0` 放到第一；
- `keyword` 在 Top 2 / Top 3 上区分能力较弱；
- `embedding` 能把与“数值计算”语义相关的相邻 chunk 拉上来。

结论：

- embedding 相比 keyword 具有更好的语义扩展能力。

---

### 9.2 Rerank 改善案例

Query：

```text
向量接近程度怎么衡量
```

第一阶段使用 keyword 检索时，结果为：

| rank | chunk_id | retrieval_score | 内容摘要 |
|---:|---|---:|---|
| 1 | `a_chunk_1` | 0.5774 | 数值计算。它支持数组和向量操作。 |
| 2 | `b_chunk_0` | 0.5774 | 余弦相似度常用于衡量两个向量方向的接近程度。 |
| 3 | `a_chunk_0` | 0.0000 | Python 是一种常用编程语言。numpy 适合数值计算。 |

在这个例子里，`a_chunk_1` 和 `b_chunk_0` 都命中了“向量”，因此 keyword baseline 给出了相同的 retrieval_score。由于第一阶段检索只基于关键词计数，它无法准确区分哪个 chunk 更适合回答“向量接近程度怎么衡量”。

接入 CrossEncoder reranker 后，排序变为：

| rank | chunk_id | retrieval_score | rerank_score | 内容摘要 |
|---:|---|---:|---:|---|
| 1 | `b_chunk_0` | 0.5774 | 0.9971 | 余弦相似度常用于衡量两个向量方向的接近程度。 |
| 2 | `a_chunk_1` | 0.5774 | 0.0274 | 数值计算。它支持数组和向量操作。 |
| 3 | `a_chunk_0` | 0.0000 | 0.0005 | Python 是一种常用编程语言。numpy 适合数值计算。 |

这一结果说明：

- retriever 负责快速召回候选结果；
- reranker 进一步判断 query 和 chunk 的匹配程度；
- 即使两个 chunk 的 retrieval_score 相同，reranker 也能根据完整语义重新排序；
- 对 RAG 来说，rerank 后更相关的 chunk 会被放在 prompt 更靠前的位置，从而改善最终上下文质量。

---

### 9.3 当前实验结论

当前项目的实验结果可以总结为：

- `keyword` 方案实现简单、可解释性强，在强关键词命中场景下表现稳定；
- `embedding` 方案在“表达变化较大但语义相近”的 query 上更有优势；
- embedding 并不总是更“干净”，小数据集下也可能带来弱相关召回；
- reranker 适合作为第二阶段排序模块，用于提升候选 chunk 的最终排序质量；
- rerank 的代价是推理速度更慢，因此不适合直接对全量 chunk 做 rerank，更适合作用于第一阶段召回出的 candidates；
- 当前更准确的表述是：“embedding 增强了语义召回能力，reranker 改善了候选结果排序质量”，而不是“某一种方法在所有场景下全面优于其他方法”。

---

## 10. RAG 模式说明

当前版本的 `rag` 模式已经可以完成：

- 根据 query 检索候选 chunks；
- 对候选 chunks 进行可选 rerank；
- 将最终 chunks 组织为结构化 prompt；
- 基于 prompt 生成回答；
- 输出引用来源。

### `mock` generator 的价值

`mock` 版虽然不调用真实 LLM，但它的意义不是“凑功能”，而是：

- 先验证 RAG 主链路是否完整；
- 让 retrieval / rerank 结果可以真正影响最终回答上下文；
- 在不依赖外部 API 的情况下稳定演示整个流程；
- 方便观察不同检索策略对最终上下文的影响。

从当前运行结果看：

- keyword baseline 可能会把弱相关 chunk 排在前面；
- embedding retrieval 能增强语义召回；
- reranker 可以进一步改善最终上下文排序；
- retrieval 质量会直接影响 generation 输入质量。

### `llm` generator 的当前状态

当前版本结构上已支持真实 LLM generator，但由于本阶段不再继续投入到：

- API 配额；
- 账单配置；
- 平台参数调试；

因此项目在实际演示与交付层面，仍以 `mock` 为主。

这个取舍是有意为之，核心目的是：

- 保证项目主链路完整；
- 控制实现复杂度；
- 把精力集中在检索、rerank 与 RAG 框架本身，而不是外部 API 运维细节。

---

## 11. 当前版本的局限

当前版本是一个具备基础工程结构的简易文档检索 / RAG 原型，但仍有一些明显限制：

1. 仅支持 `.txt` 文件；
2. 仅支持固定长度 chunking；
3. 数据集很小，评估仍以人工观察为主；
4. `mock` generator 不是正式 LLM 回答，只是用于打通和演示最小 RAG 闭环；
5. `llm` generator 虽然结构上已经预留，但当前未作为主完成项推进；
6. 还没有接入 FAISS、Qdrant、Milvus 等向量索引或向量数据库；
7. 还没有持久化索引，每次运行仍会重新构建 chunk embeddings；
8. 还没有系统化评估 RAG 回答质量；
9. CrossEncoder reranker 会带来额外推理开销，当前尚未统计延迟指标；
10. 当前只验证了少量样例，仍需要进一步构建 evaluation set。

---

## 12. 后续优化方向

后续可以继续沿以下方向推进。

### 方向 1：增加检索评估

优先级最高。

计划增加：

- `eval/questions.json`
- Top1 Hit
- Recall@K
- MRR
- 平均检索耗时
- rerank 前后结果对比

目标是回答：

> rerank 是否真的提升了检索排序质量？提升了多少？代价是多少？

---

### 方向 2：补充错误案例分析

针对检索失败或 rerank 失败的 case，记录：

- query；
- 原始 top-k 结果；
- rerank 后结果；
- 错误原因；
- 是否由于 chunking、召回不足、模型语义误判或数据过少导致。

这部分比继续堆功能更有面试价值。

---

### 方向 3：优化 chunking 策略

例如：

- 按句子切分；
- 按段落切分；
- 基于 token 数切分；
- 更合理的 overlap 策略。

当前固定长度 chunking 实现简单，但可能切断语义边界。

---

### 方向 4：引入向量索引或向量数据库

当数据规模进一步增大后，可继续尝试：

- FAISS：轻量本地向量索引，适合替代 numpy 暴力相似度计算；
- Qdrant：适合体验服务化向量检索和 metadata filtering；
- Milvus：更偏生产级和大规模场景，当前阶段优先级较低。

当前项目阶段不急于接入向量数据库，因为主要目标是先把 retrieval、rerank、evaluation 和 error analysis 做清楚。

---

### 方向 5：继续推进真实生成

在已有 `llm` generator 预留基础上，后续如果需要，可以继续完善：

- API 配置方式；
- 真实回答调用；
- 生成质量评估；
- 引用与回答联动。

---

## 13. 面试讲法

可以用下面这段概括项目：

> 这个项目是一个轻量级 RAG 文档检索原型。我没有一开始直接使用 LangChain，而是从底层实现了文档读取、chunk 切分、metadata 管理、keyword / embedding 检索、chunk embedding 预计算、prompt 构造和 mock generator。后续我发现第一阶段检索虽然能召回相关 chunk，但不一定能把最适合回答问题的 chunk 排在最前面，所以进一步引入了二阶段 rerank。当前支持 NoOpReranker 作为 baseline，也支持 CrossEncoder / BGE reranker 对候选 chunk 重新打分，并输出 retrieval_score 和 rerank_score 来分析 rerank 前后的排序变化。

如果被问“为什么不用 LangChain”，可以回答：

> 这个项目的目标不是快速搭一个黑盒应用，而是理解 RAG 主链路中每个模块的作用和取舍。自己实现 splitter、retriever、reranker、prompt builder 后，更容易解释 chunking、embedding、top-k retrieval、rerank、上下文构造这些关键问题。后续如果接入 LangChain，也能清楚知道它封装了哪些环节，而不是只会调用接口。

如果被问“为什么不直接接向量数据库”，可以回答：

> 当前数据规模很小，核心问题不是向量存储能力，而是检索质量、排序质量和评估方式。因此我先用本地 embedding + cosine similarity 打通链路，再加入 rerank 和评估。等数据规模变大或需要 metadata filtering、持久化、服务化检索时，再接 FAISS 或 Qdrant 会更合理。

如果被问“rerank 有什么作用”，可以回答：

> 第一阶段 retriever 更适合快速召回候选，但排序不一定最准确。reranker 会把 query 和候选 chunk 一起输入模型，重新判断相关性。比如在“向量接近程度怎么衡量”这个例子里，keyword 检索把“支持数组和向量操作”和“余弦相似度衡量向量接近程度”打成同分，但 reranker 能把真正回答问题的 chunk 提升到第一位。

---

## 14. 项目总结

这个项目当前已经从一个“最小可运行检索 demo”升级为一个具备基础工程结构的简易文档检索 / RAG 原型，完成了：

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
- 命令行调用；
- 基础测试；
- 初步效果对比与分析。

当前版本虽然仍然轻量，但已经具备：

- 清晰的演进路径；
- 可运行的演示方式；
- 可讲解的项目叙事；
- 可扩展的模块结构；
- 面向后续 evaluation、FAISS/Qdrant、真实 LLM 接入的基础。

下一步最值得做的不是继续堆新组件，而是补充 evaluation 和 error analysis，让项目能够回答：

> 检索效果有没有提升？为什么提升？在哪些 case 失败？工程上如何取舍？
