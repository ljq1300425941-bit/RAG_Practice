# RAG 检索与 Rerank 错误案例分析

本文档用于记录当前 RAG 原型中的检索成功案例、失败案例和 rerank 取舍分析。

当前实验设置：

- 数据规模：5 个 chunks；
- 评估 query 数量：5 条；
- `retrieve_top_k=5`；
- `rerank_top_n=3`；
- embedding model：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`；
- reranker model：`BAAI/bge-reranker-base`。

当前四组实验 summary：

| 方法 | Top1 Hit | Recall@3 | MRR | Avg Latency |
|---|---:|---:|---:|---:|
| keyword only | 0.8000 | 1.0000 | 0.9000 | 0.03ms |
| keyword + rerank | 0.8000 | 1.0000 | 0.9000 | 154.12ms |
| embedding only | 0.8000 | 1.0000 | 0.9000 | 28.76ms |
| embedding + rerank | 0.8000 | 1.0000 | 0.9000 | 163.30ms |

当前总体结论：

> rerank 能改善部分候选排序问题，但在当前小数据集上没有带来整体指标提升，并且会引入明显延迟。它不是无条件提升准确率的模块，需要通过 evaluation 和 error analysis 判断是否值得使用。

---

## Case 1：keyword 同分，rerank 成功修正排序

### Query

```text
两个向量相不相近一般看什么指标
```

### Expected

```text
b_chunk_0
```

`b_chunk_0` 内容与“向量相近程度如何衡量”相关，包含“余弦相似度”“向量方向的接近程度”等关键信息。

---

### keyword only 结果

| rank | chunk_id | retrieval_score | rerank_score |
|---:|---|---:|---|
| 1 | `a_chunk_1` | 0.5774 | None |
| 2 | `b_chunk_0` | 0.5774 | None |
| 3 | `a_chunk_0` | 0.0000 | None |

### 现象

`a_chunk_1` 和 `b_chunk_0` 的 `retrieval_score` 相同，keyword baseline 无法区分二者。

原因是二者都命中了“向量”相关关键词：

- `a_chunk_1`：只是在说“支持数组和向量操作”；
- `b_chunk_0`：真正解释“余弦相似度用于衡量向量接近程度”。

keyword retriever 只能基于词频做粗粒度匹配，因此会出现“命中词相同，但语义相关性不同”的问题。

---

### keyword + rerank 结果

| rank | chunk_id | retrieval_score | rerank_score |
|---:|---|---:|---:|
| 1 | `b_chunk_0` | 0.5774 | 0.9133 |
| 2 | `a_chunk_1` | 0.5774 | 0.0322 |
| 3 | `c_chunk_0` | 0.0000 | 0.0001 |

### 分析

reranker 将 query 和 chunk 一起输入模型，直接判断二者的相关性。

在这个 case 中，reranker 识别到：

```text
两个向量相不相近一般看什么指标
```

与：

```text
余弦相似度常用于衡量两个向量方向的接近程度
```

语义更接近，因此将 `b_chunk_0` 提升到第一位。

### 结论

这是一个 rerank 有效的典型案例：

- 第一阶段 retriever 成功召回了正确 chunk；
- 但原始排序不理想；
- reranker 在候选集合内部修正了排序。

这个案例说明：

> rerank 的主要价值不是扩大召回范围，而是在已召回候选中改善排序质量。

---

## Case 2：rerank 引入新的排序错误

### Query

```text
怎么判断检索结果和问题是否相关
```

### Expected

```text
b_chunk_0
```

当前标注中，`b_chunk_0` 被认为是正确 chunk，因为它与“相似度”“检索任务中衡量相关性”更相关。

---

### keyword only 结果

| rank | chunk_id | retrieval_score | rerank_score |
|---:|---|---:|---|
| 1 | `b_chunk_0` | 0.5774 | None |
| 2 | `a_chunk_0` | 0.0000 | None |
| 3 | `a_chunk_1` | 0.0000 | None |

### 现象

keyword baseline 在这个 query 上表现正确，`b_chunk_0` 排在第一。

---

### keyword + rerank 结果

| rank | chunk_id | retrieval_score | rerank_score |
|---:|---|---:|---:|
| 1 | `c_chunk_0` | 0.0000 | 0.0619 |
| 2 | `b_chunk_0` | 0.5774 | 0.0595 |
| 3 | `b_chunk_1` | 0.0000 | 0.0062 |

### 现象

reranker 将 `c_chunk_0` 提升到了第一，而 `b_chunk_0` 下降到第二。

这导致：

```text
Top1Hit: 1 → 0
RR: 1.0000 → 0.5000
```

---

### 原因分析

这个 case 有几个可能原因：

#### 1. chunk 内容太短，模型判断空间有限

当前数据集只有 5 个 chunks，且 chunk 内容较短。短文本可能缺少足够上下文，reranker 容易根据局部词或抽象语义误判。

#### 2. query 表达比较抽象

```text
怎么判断检索结果和问题是否相关
```

这个 query 并没有明确出现“余弦相似度”“向量”等词，而是在问“相关性判断”这个更抽象的问题。

对于当前极小数据集来说，这种 query 可能让 reranker 对不同 chunk 的分数区分不稳定。

#### 3. rerank 分数差距很小

`c_chunk_0` 与 `b_chunk_0` 的 rerank 分数分别是：

```text
c_chunk_0: 0.0619
b_chunk_0: 0.0595
```

差距非常小，说明 reranker 本身也没有很强的置信度。

如果在工程上处理这类 case，可以考虑：

- 当 rerank 分数差距很小时，保留原始 retrieval_score 作为 tie-break；
- 使用 retrieval_score 和 rerank_score 的加权融合；
- 增加 minimum confidence threshold；
- 扩大评估集观察是否是偶然误差。

---

### 结论

这是一个 rerank 失败案例。

它说明：

> reranker 不是无条件可靠的。它可能修正 baseline 错误，也可能在低置信度场景下引入新的排序错误。

---

## Case 3：embedding 能增强语义召回，但不一定改善 Top1

### Query

```text
怎么判断检索结果和问题是否相关
```

### Expected

```text
b_chunk_0
```

### embedding only 结果

| rank | chunk_id | retrieval_score | rerank_score |
|---:|---|---:|---|
| 1 | `b_chunk_1` | 0.5784 | None |
| 2 | `b_chunk_0` | 0.4645 | None |
| 3 | `a_chunk_1` | 0.3661 | None |

### 现象

embedding retriever 将 `b_chunk_1` 排到了第一，而标注的正确 chunk 是 `b_chunk_0`。

不过 `b_chunk_0` 仍然进入了 top-3，因此：

```text
Top1Hit = 0
Recall@3 = 1
RR = 0.5000
```

### 分析

这说明 embedding retrieval 具备较强召回能力，但不一定能保证正确 chunk 排在第一。

它的特点是：

- 能召回语义相关内容；
- 但可能把相邻或弱相关 chunk 放到更前面；
- 在小数据集和短 chunk 下，这种排序误差更明显。

### 结论

embedding 的价值主要体现在增强语义召回，而不是保证最终排序一定最优。

这也说明为什么真实 RAG 系统常采用：

```text
embedding retrieval 召回候选
  → rerank 重新排序
```

但是否值得 rerank，仍需要结合评估结果和延迟成本判断。

---

## Case 4：rerank 没有改善 embedding 的失败样例

### Query

```text
怎么判断检索结果和问题是否相关
```

### Expected

```text
b_chunk_0
```

### embedding + rerank 结果

| rank | chunk_id | retrieval_score | rerank_score |
|---:|---|---:|---:|
| 1 | `c_chunk_0` | 0.3623 | 0.0619 |
| 2 | `b_chunk_0` | 0.4645 | 0.0595 |
| 3 | `b_chunk_1` | 0.5784 | 0.0062 |

### 现象

embedding only 时，`b_chunk_0` 排第二；embedding + rerank 后，`b_chunk_0` 仍然排第二，而且不相关的 `c_chunk_0` 被提升到了第一。

### 分析

该 case 和 keyword + rerank 的失败类似，说明失败并不是 keyword retriever 独有问题，而可能来自 reranker 在该 query 上的低置信度判断。

其中一个关键细节是：

```text
c_chunk_0 rerank_score = 0.0619
b_chunk_0 rerank_score = 0.0595
```

分差极小。

这意味着可以考虑改进排序策略：

```text
final_score = alpha * rerank_score + (1 - alpha) * normalized_retrieval_score
```

或者当分差低于阈值时，不完全覆盖原始 retrieval 排序。

### 结论

当前 rerank 策略是“完全按照 rerank_score 排序”。这种做法简单直接，但也可能在 reranker 分数差距很小时引入错误。

后续可以尝试：

- rerank_score + retrieval_score 加权融合；
- 分数差距过小时保留原始 retrieval 排序；
- 增大评估集，看该失败是否稳定出现；
- 使用更长、更完整的 chunk 提供更多上下文。

---

## 当前阶段结论

基于目前评估结果，当前最准确的结论是：

1. 第一阶段 retrieval 的 Recall@3 已经较好，正确 chunk 基本能进入候选集；
2. rerank 能修正部分排序错误；
3. rerank 也可能引入新的排序错误；
4. rerank 带来明显延迟；
5. 当前小数据集不足以证明 rerank 整体显著提升；
6. 后续应扩大 evaluation set，并继续记录 error cases。

---

## 后续改进建议

### 1. 扩大评估集

当前只有 5 条 query，不足以稳定判断 rerank 效果。

建议扩展到至少 20 条：

- 5 条强关键词 query；
- 5 条语义改写 query；
- 5 条干扰 query；
- 5 条多答案 query。

---

### 2. 增加更多真实文档

当前只有 5 个 chunks，实验结果容易受个别 chunk 影响。

可以加入：

- OS / 网络 / 数据库笔记；
- RAG / embedding / vector database 学习笔记；
- 项目 README / 实验记录。

---

### 3. 尝试分数融合策略

当前排序策略是：

```text
sort by rerank_score
```

后续可以尝试：

```text
sort by alpha * rerank_score + (1 - alpha) * retrieval_score
```

或者：

```text
rerank_score 分差较小时，保留原 retrieval 排序
```

---

### 4. 记录更多失败案例

每次 rerank 改善或变差，都记录：

- query；
- expected chunk；
- baseline 排名；
- rerank 后排名；
- retrieval_score；
- rerank_score；
- 是否成功；
- 原因分析；
- 后续改进方向。

这样项目会从“功能 demo”变成“有实验分析的小型工程项目”。
