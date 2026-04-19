# 简易文档检索原型

一个基于 Python 实现的简易文档检索原型，支持从目录中读取多个 `.txt` 文本文件，进行 chunk 切分、简化向量化、余弦相似度计算，并返回 top-k 检索结果。

这个项目当前的重点不是复杂框架，而是先完成一条清晰、可运行、可扩展的最小检索闭环：

> 文档读取 → chunk 切分 → 向量化 → 相似度计算 → top-k 检索

---

## 1. 项目功能

当前版本支持：

- 读取指定目录下的多个 `.txt` 文件
- 将文本按固定长度切分为多个 chunk
- 为每个 chunk 保存基础 metadata
- 使用简化关键词计数方式将文本映射为向量
- 使用余弦相似度计算 query 与各个 chunk 的相关性
- 返回 top-k 检索结果
- 通过命令行参数指定输入目录、query、chunk 参数和 top-k

---

## 2. 项目结构

```text
PythonPractice/
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

用于表示切分后的文本块，包含：

- `chunk_id`
- `source_file`
- `text`
- `start_pos`
- `end_pos`

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
负责将文本转换为简化向量表示。

当前版本使用的是**关键词计数向量**，本质上是一个非常简化的 bag-of-words 方案。  
例如，给定一个词表：

```python
["python", "numpy", "向量", "相似度", "数据库", "检索"]
```

文本中每个词出现的次数会组成一个固定长度向量。

这一方案语义能力有限，但适合在项目第一版中先跑通完整检索流程。

---

### `retriever.py`
负责检索逻辑：

- 计算余弦相似度
- 遍历所有 chunk
- 计算 query 与 chunk 的相关性分数
- 排序并返回 top-k 结果

---

### `cli.py`
负责命令行参数解析。

当前支持参数：

- `--input_dir`
- `--query`
- `--chunk_size`
- `--overlap`
- `--top_k`

---

### `main.py`
项目入口文件，负责串联完整流程：

1. 读取命令行参数
2. 构建 chunks
3. 执行检索
4. 打印结果

---

## 4. 核心流程

项目当前的整体流程如下：

```text
输入文档目录
   ↓
读取多个 .txt 文件
   ↓
文本切分为 chunks
   ↓
为每个 chunk 构建 metadata
   ↓
将 query 和 chunk 文本映射为简化向量
   ↓
使用余弦相似度计算相关性
   ↓
按分数排序，返回 top-k
```

---

## 5. 环境依赖

### 安装依赖

```bash
pip install -r requirements.txt
```

当前 `requirements.txt` 可包含：

```text
numpy
pytest
```

---

## 6. 运行方式

在项目根目录下运行：

```bash
python main.py --input_dir data --query "numpy 向量 相似度" --chunk_size 30 --overlap 5 --top_k 3
```

参数说明：

- `input_dir`：输入文档目录
- `query`：检索查询
- `chunk_size`：每个 chunk 的长度
- `overlap`：相邻 chunk 的重叠长度
- `top_k`：返回前 k 个最相关结果

---

## 7. 示例输出

下面是一组实际运行结果示例：

```text
共构建 5 个 chunks

开始检索...

[Top 1] score=0.6667
chunk_id=b_chunk_0
source_file=b.txt
text=余弦相似度常用于衡量两个向量方向的接近程度。在检索任务中可以
----------------------------------------
[Top 2] score=0.4082
chunk_id=a_chunk_0
source_file=a.txt
text=Python 是一种常用编程语言。numpy 适合数值计算。
----------------------------------------
[Top 3] score=0.4082
chunk_id=a_chunk_1
source_file=a.txt
text=数值计算。它支持数组和向量操作。
----------------------------------------
```

从结果可以看出：

- 包含“相似度”“向量”“检索”等关键词的 chunk 排名更靠前
- 与 query 语义无关的数据库文本没有进入前 3
- 当前简化检索流程能够输出基本合理的结果

---

## 8. 测试

在项目根目录运行：

```bash
pytest
```

当前测试结果：

```text
================================================================================= test session starts ==================================================================================
platform win32 -- Python 3.14.4, pytest-9.0.3, pluggy-1.6.0
rootdir: D:\project\PythonPractice
configfile: pytest.ini
testpaths: tests
collected 5 items

tests\test_retriever.py ...                                                                                                                                                       [ 60%]
tests\test_splitter.py ..                                                                                                                                                         [100%]

================================================================================== 5 passed in 0.05s ===================================================================================
```

当前建议至少包含以下测试：

- `test_splitter.py`
- `test_retriever.py`

测试重点包括：

### `splitter`
- 正常切分
- 空文本处理
- `chunk_size <= 0`
- `overlap >= chunk_size`

### `retriever`
- 正常返回 top-k
- 返回结果按分数降序排列
- `k` 大于结果总数时的行为
- 零向量场景下程序不崩溃

---

## 9. 当前版本的局限

当前版本是一个**最小可运行原型**，主要用于验证检索链路，仍有一些明显限制：

1. 仅支持 `.txt` 文件
2. 仅支持固定长度 chunking
3. 向量化方式较为简单，只能做粗粒度关键词匹配
4. 不具备真实 embedding 的语义表示能力
5. 不支持向量数据库和更高效的大规模检索
6. 还没有接入生成式回答模块

这些限制是当前版本有意保留的，目的是先保证主链路清晰可控。

---

## 10. 后续优化方向

后续可以逐步升级到更完整的版本，例如：

### 方向 1：接入真实 embedding
将当前关键词计数向量替换为真实 embedding 表示，例如：

- sentence-transformers
- OpenAI embedding
- 其他本地向量模型

这样可以显著提升语义检索能力。

---

### 方向 2：支持更多文档格式
未来可扩展支持：

- `.md`
- `.json`
- `.pdf`

进一步提升项目实用性。

---

### 方向 3：优化 chunking 策略
当前使用的是固定长度切分，后续可以尝试：

- 按句子切分
- 按段落切分
- 基于 token 数切分
- 更合理的 overlap 策略

---

### 方向 4：补全 RAG 闭环
在检索结果基础上，进一步实现：

- 检索结果拼接
- 构造 prompt
- 接入 LLM 生成回答

从“文档检索原型”升级到“简易 RAG 原型”。

---

## 11. 项目总结

这个项目当前版本实现了一个完整的最小文档检索闭环，包括：

- 文档读取
- 文本切分
- metadata 管理
- 简化向量化
- 余弦相似度计算
- top-k 检索
- 命令行调用
- 基础测试支持

虽然当前版本仍然较轻量，但结构清晰，便于继续扩展到更真实的 embedding 检索与 RAG 场景。
