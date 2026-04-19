# rag_pipeline

模块化 RAG（检索增强生成）流水线，整合分块、向量化、存储、检索、生成五大模块。

## 架构

```
[ magic_chunker ]  →  [ magic_embedder ]  →  [ magic_vectorstore ]  →  [ magic_retriever ]  →  [ magic_generator ]
        分块                 向量化                   向量存储                   检索                  生成
```

## 包含模块

| 模块 | 功能 | 策略 |
|------|------|------|
| [magic_chunker](https://github.com/seiner69/magic_chunker) | 文档分块 | SemanticChunker、ParentChildChunker |
| [magic_embedder](https://github.com/seiner69/magic_embedder) | 文本/图像向量化 | OpenAI、SentenceTransformer、CLIP |
| [magic_vectorstore](https://github.com/seiner69/magic_vectorstore) | 向量存储检索 | ChromaDB、FAISS、InMemoryStore |
| [magic_retriever](https://github.com/seiner69/magic_retriever) | 相似度/MMR 检索 | SimilarityRetriever、MMRRetriever、ParentChildRetriever |
| [magic_generator](https://github.com/seiner69/magic_generator) | LLM 生成 | OpenAI GPT、Anthropic Claude |

## 切块策略

### 语义切块 (默认)

按段落和 Markdown 标题自然边界切分，适合通用文档。

```python
config = RAGPipelineConfig(chunking_strategy="semantic")
```

### 父子文档切块 (适合金融财报)

专为高度结构化文档设计（如茅台年报），解决检索精度与上下文完整性的矛盾。

**核心思想**：
- **父文档**：较大块 (默认 1000 字符)，保持完整上下文
- **子文档**：较小块 (默认 200 字符)，用于精准向量检索，带 overlap
- **分离存储**：子文档 → FAISS 向量库，父文档 → InMemoryStore 键值存储
- **检索时**：先命中子文档，再通过 `parent_id` 召回完整父文档

```python
config = RAGPipelineConfig(
    chunking_strategy="parent_child",
    parent_chunk_size=1000,   # 父文档大小
    parent_overlap=100,        # 父文档 overlap
    child_chunk_size=200,      # 子文档大小
    child_overlap=50,          # 子文档 overlap
    vectorstore_type="faiss",
    faiss_index_type="hnsw",   # HNSW 索引，高召回
)
```

**CLI 用法**：

```bash
# 父子文档模式索引
python -m rag_pipeline.run \
    --action chunk \
    --chunking parent_child \
    --parent-size 1000 \
    --child-size 200 \
    --files /path/to/content_list.json

# 完整流水线
python -m rag_pipeline.run \
    --action run \
    --chunking parent_child \
    --vectorstore faiss \
    --faiss-index hnsw \
    --query "茅台2024年营收是多少？"
```

## 快速开始

### Python API

```python
from rag_pipeline import RAGPipeline, RAGPipelineConfig

# 普通模式
config = RAGPipelineConfig(
    embedder_type="sentence_transformer",
    vectorstore_type="chroma",
)

# 父子文档模式 (金融财报)
config = RAGPipelineConfig(
    chunking_strategy="parent_child",
    parent_chunk_size=1000,
    child_chunk_size=200,
    vectorstore_type="faiss",
    faiss_index_type="hnsw",
)

pipeline = RAGPipeline(config)
pipeline.chunk(files=["/path/to/content_list.json"])
pipeline.embed()
pipeline.store()
result = pipeline.run("公司营收是多少？")
print(result)
```

### CLI

```bash
# 检索
python -m rag_pipeline.run \
    --action retrieve \
    --query "公司营收是多少？"

# 父子文档模式
python -m rag_pipeline.run \
    --action run \
    --chunking parent_child \
    --parent-size 1000 \
    --child-size 200 \
    --vectorstore faiss \
    --faiss-index hnsw \
    --query "茅台2024年营收是多少？"
```

## 支持的组件

### 向量化 (embedder_type)

| 类型 | 模型 |
|------|------|
| `sentence_transformer` | all-MiniLM-L6-v2, all-mpnet-base-v2 |
| `openai` | text-embedding-3-small, text-embedding-3-large, ada-002 |
| `clip` | openai/clip-vit-base-patch32 |

### 向量存储 (vectorstore_type)

| 类型 | 说明 |
|------|------|
| `chroma` | ChromaDB，持久化/内存模式 |
| `faiss` | FAISS，flat/ivf/hnsw 索引 |
| `InMemoryStore` | 内存键值存储，用于父文档 |

### FAISS 索引类型 (faiss_index_type)

| 类型 | 说明 | 适用场景 |
|------|------|---------|
| `flat` | 暴力精确搜索 | 小数据量 (<1万) |
| `hnsw` | 分层图导航小世界 | 大规模，高召回 |
| `ivf` | 倒排索引，聚类搜索 | 中等规模 |

### 检索 (retriever_type)

| 类型 | 说明 |
|------|------|
| `similarity` | 相似度检索，余弦相似度 |
| `mmr` | 最大边际相关性，平衡相关性与多样性 |
| `parent_child` | 父子文档检索，自动召回父文档 |

### 生成 (generator_type)

| 类型 | 模型 |
|------|------|
| `openai` | gpt-4o, gpt-4o-mini, gpt-4-turbo |
| `anthropic` | claude-sonnet, claude-haiku, claude-opus |

## 安装依赖

```bash
pip install sentence-transformers chromadb faiss-cpu openai anthropic transformers torch torchvision Pillow
```

## 模块结构

```
rag_pipeline/
    __init__.py
    run.py               # CLI 入口
    pipeline.py          # 流水线主类
    magic_chunker/       # 分块模块
    magic_embedder/      # 向量化模块
    magic_vectorstore/   # 向量存储模块
    magic_retriever/     # 检索模块
    magic_generator/     # 生成模块
```
