# 简易文档检索原型（README V2）

一个基于 Python 实现的简易文档检索原型，支持从目录中读取多个 `.txt` 文本文件，进行 chunk 切分、向量化、相似度计算，并返回 top-k 检索结果。

当前版本的重点是逐步从“最小可运行检索 demo”升级到“更接近真实语义检索系统的原型”，整体演进路线如下：

> 文档读取 → chunk 切分 → metadata 管理 → 向量化 → 相似度计算 → top-k 检索 → embedding 升级 → chunk embedding 预计算

---

## 1. 项目功能

当前版本支持：

- 读取指定目录下的多个 `.txt` 文件
- 将文本按固定长度切分为多个 chunk
- 为每个 chunk 保存基础 metadata
- 支持两种向量化方式：
  - `keyword`：关键词计数向量
  - `embedding`：真实 embedding 向量
- 使用余弦相似度计算 query 与 chunk 的相关性
- 返回 top-k 检索结果
- 支持命令行参数指定输入目录、query、chunk 参数、top-k 和向量化模式
- 对 chunk embeddings 做预计算，避免查询时重复编码全部文档
- 提供基础测试，验证 splitter 与 retriever 主流程

---

## 2. 项目结构

```text
RAG_Practice/
  main.py
  app/
    __init__.py
    models.py
    loader.py
    splitter.py
    vectorizer.py
    retriever.py
    cli.py
  data/
    a.txt
    b.txt
    c.txt
  tests/
    test_splitter.py
    test_retriever.py
  requirements.txt
  README.md
```

---

## 3. 核心模块说明

### `models.py`
定义项目中的核心数据结构：

- `DocumentChunk`
- `ChunkEmbedding`

其中：

- `DocumentChunk` 用于表示切分后的文本块，包含：
  - `chunk_id`
  - `source_file`
  - `text`
  - `start_pos`
  - `end_pos`

- `ChunkEmbedding` 用于表示：
  - 一个 `DocumentChunk`
  - 它对应的向量表示 `embedding`

这样可以避免在代码中维护容易错位的 parallel list。

---

### `loader.py`
负责文件读取：

- 扫描输入目录下的 `.txt` 文件
- 返回文件路径列表

---

### `splitter.py`
负责文本切分与 chunk 构建：

- 将原始文本按 `chunk_size` 和 `overlap` 切分
- 构建 `DocumentChunk`
- 支持从目录批量构建 chunk 列表

---

### `vectorizer.py`
负责文本转向量。

当前版本支持两种模式：

#### 1）`KeywordCountVectorizer`
使用关键词计数向量，本质上是一个非常简化的 bag-of-words 方案。

例如给定词表：

```python
["python", "numpy", "向量", "相似度", "数据库", "检索"]
```

文本中每个词出现的次数会组成一个固定长度向量。

优点：

- 实现简单
- 易于理解
- 在强关键词命中场景下表现直观

局限：

- 更依赖字面词匹配
- 对表达变化和语义相近表达不敏感

#### 2）`EmbeddingVectorizer`
使用真实 embedding 模型对文本进行编码。

当前实验中引入了多语言 sentence-transformers 模型，用于提升中文场景下的语义检索能力。

优点：

- 对语义表达变化更敏感
- 能处理不完全同词的查询与文本匹配

局限：

- 编码成本更高
- 小数据集上仍可能引入弱相关噪声结果

---

### `retriever.py`
负责检索逻辑：

- 计算余弦相似度
- 构建 chunk embeddings
- 对 query 编码
- 基于预计算好的 chunk embeddings 做 top-k 检索

当前版本的检索流程已经从“每次查询现场编码全部 chunk”升级为：

- 文档加载后先编码全部 chunks
- 查询时只编码 query
- 再与已有 chunk embeddings 计算相似度

这样更接近真实检索系统的工作方式。

---

### `cli.py`
负责命令行参数解析。

当前支持参数：

- `--input_dir`
- `--query`
- `--chunk_size`
- `--overlap`
- `--top_k`
- `--vectorizer`
- `--model_name`

---

### `main.py`
项目入口文件，负责串联完整流程：

1. 读取命令行参数
2. 构建 chunks
3. 选择 vectorizer
4. 预计算 chunk embeddings
5. 执行检索
6. 打印结果

---

## 4. 项目演进过程

### V1：最小检索闭环
第一阶段实现了一个最小可运行的检索原型，包括：

- 文档读取
- chunk 切分
- metadata 管理
- 关键词计数向量
- 余弦相似度计算
- top-k 检索
- 命令行运行
- 基础测试

这一阶段的目标不是追求高语义能力，而是先把检索链路跑通。

---

### V2：接入真实 embedding
第二阶段在保留 `keyword` 模式的基础上，新增了 `embedding` 模式。

最初接入英文向模型时，在中文 query 场景下结果不够稳定，因此进一步切换到更适合多语言场景的 embedding 模型，中文检索结果明显更合理。

这一阶段的重点是：

- 保留 V1 关键词方案，作为 baseline
- 引入真实 embedding，提升语义检索能力
- 做 keyword / embedding 对比分析

---

### V2.1：增加 chunk embedding 预计算
在 embedding 模式下，如果每次 query 都重新编码所有 chunks，会带来明显重复计算。

因此在当前版本中进一步加入了：

- 文档 chunk embeddings 预计算
- 查询时仅编码 query
- 检索阶段直接复用已有 chunk embeddings

这一升级使得项目流程更接近真实检索系统。

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
余弦相似度计算
   ↓
按分数排序，返回 top-k
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
```

---

## 7. 运行方式

### keyword 模式

```bash
python main.py --input_dir data --query "numpy 向量 相似度" --chunk_size 30 --overlap 5 --top_k 3 --vectorizer keyword
```

### embedding 模式

```bash
python main.py --input_dir data --query "向量接近程度怎么衡量" --chunk_size 30 --overlap 5 --top_k 3 --vectorizer embedding --model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

参数说明：

- `input_dir`：输入文档目录
- `query`：检索查询
- `chunk_size`：每个 chunk 的长度
- `overlap`：相邻 chunk 的重叠长度
- `top_k`：返回前 k 个最相关结果
- `vectorizer`：向量化方式，支持 `keyword` / `embedding`
- `model_name`：embedding 模型名称，仅 embedding 模式生效

---

## 8. 测试

在项目根目录运行：

```bash
pytest
```

当前测试已通过，覆盖 splitter 与 retriever 的基础主流程。

---

## 9. 效果对比与分析

基于 4 组 query，对 `keyword` 与 `embedding` 两种模式做了对比。

### Query 1：`numpy 向量 相似度`

- `keyword` 与 `embedding` 都能返回较合理的结果
- 两者的 Top 1 都是与“相似度 / 向量”最相关的 chunk
- 这一组更像是在说明：当 query 的关键词非常明确时，keyword baseline 本身也能工作得不错

结论：

- 强关键词场景下，两者差距不大
- embedding 没有显著拉开优势，但也没有破坏结果

---

### Query 2：`向量接近程度怎么衡量`

- `keyword` 更依赖字面词匹配，把“支持数组和向量操作”的 chunk 排到了第一
- `embedding` 能把“接近程度怎么衡量”与“余弦相似度常用于衡量两个向量方向的接近程度”对应起来
- 这一组最能体现语义检索的优势

结论：

- 在表达变化较大但语义相关的 query 上，embedding 明显更合理

---

### Query 3：`数据库 事务`

- 两种模式都能把数据库相关 chunk 排到第一
- `keyword` 后续结果更“硬匹配”，很多非相关项分数直接为 0
- `embedding` 在后续结果中引入了一些弱相关内容

结论：

- embedding 不一定在所有场景下都更“干净”
- 在小数据集里，embedding 可能会带来更宽松的召回，也会夹带一定噪声

---

### Query 4：`Python 数值计算`

- 两种模式都能正确把 `a_chunk_0` 放到第一
- `keyword` 在 Top 2 / Top 3 上区分能力较弱
- `embedding` 能把与“数值计算”语义相关的相邻 chunk 拉上来

结论：

- embedding 相比 keyword 具有更好的语义扩展能力

---

### 总体结论

当前项目的对比结果可以总结为：

- `keyword` 方案实现简单、可解释性强，在强关键词命中场景下表现稳定
- `embedding` 方案在“表达变化较大但语义相近”的 query 上更有优势
- 在当前小数据集上，embedding 结果已经比纯关键词方案更像真正的语义检索，但仍会出现一些弱相关噪声
- 因此当前更准确的表述是：“embedding 版增强了语义检索能力”，而不是“在所有场景下全面优于 keyword 方案”

---

## 10. 当前版本的局限

当前版本是一个具备基础工程结构的简易语义检索原型，但仍有一些明显限制：

1. 仅支持 `.txt` 文件
2. 仅支持固定长度 chunking
3. 数据集很小，评估仍以人工观察为主
4. 目前只做了 top-k 检索，还没有接入 answer generation
5. 还没有接入向量数据库或持久化索引
6. embedding 模式虽已初步可用，但模型选型和检索评估仍可进一步优化

---

## 11. 后续优化方向

后续可以继续沿以下方向推进：

### 方向 1：支持更多文档格式
例如：

- `.md`
- `.json`
- `.pdf`

### 方向 2：优化 chunking 策略
例如：

- 按句子切分
- 按段落切分
- 基于 token 数切分
- 更合理的 overlap 策略

### 方向 3：增加检索评估
例如：

- 扩充 query 集合
- 构建更系统的人工标注
- 统计不同模式下的命中质量

### 方向 4：接入 RAG 生成回答
在检索结果基础上进一步实现：

- 检索结果拼接
- prompt 构造
- 接入 LLM 回答生成

### 方向 5：引入向量索引或向量数据库
当数据规模进一步增大后，可继续尝试：

- Faiss
- Chroma
- 其他向量检索组件

---

## 12. 项目总结

这个项目当前已经从一个“最小可运行检索 demo”升级为一个具备基础工程结构的简易语义检索原型，完成了：

- 文档读取
- 文本切分
- metadata 管理
- keyword / embedding 双模式向量化
- 余弦相似度计算
- top-k 检索
- chunk embedding 预计算
- 命令行调用
- 基础测试
- 初步效果对比与分析

当前版本虽然仍然轻量，但已经具备清晰的演进路径，也为后续升级到更完整的 RAG 原型打下了基础。
