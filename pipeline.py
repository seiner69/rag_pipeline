"""
RAG 流水线主模块
支持两种模式：
1. 普通模式：单层切块 → 向量化 → 存储 → 检索 → 生成
2. 父子文档模式：双层切块 → 子向量检索 → 父文档召回 → 去重 → 生成
"""

from dataclasses import dataclass, field
from typing import Optional

from magic_chunker.core import ChunkingResult
from magic_embedder.core import EmbeddingResult
from magic_vectorstore.core import QueryResult
from magic_retriever.core import RetrievalResult
from magic_generator.core import GenerationPrompt, GenerationResult


@dataclass
class RAGPipelineConfig:
    """RAG 流水线配置"""

    # ============== 通用配置 ==============
    # 切块策略: "semantic" | "parent_child"
    chunking_strategy: str = "semantic"

    # 向量化配置
    embedder_type: str = "sentence_transformer"  # openai | sentence_transformer | clip
    embedder_model: str = "all-MiniLM-L6-v2"

    # 向量存储配置
    vectorstore_type: str = "chroma"  # chroma | faiss
    collection_name: str = "rag_collection"
    persist_dir: Optional[str] = "./chroma_db"

    # FAISS 索引类型: "flat" | "hnsw" | "ivf"
    faiss_index_type: str = "hnsw"

    # 检索配置
    retriever_type: str = "similarity"  # similarity | mmr
    top_k: int = 5
    fetch_k: int = 20
    lambda_mult: float = 0.5  # MMR 参数

    # 生成配置
    generator_type: str = "openai"  # openai | anthropic
    generator_model: str = "gpt-4o-mini"

    # ============== 父子文档模式配置 ==============
    # 父文档切块大小（字符数）
    parent_chunk_size: int = 1000
    parent_overlap: int = 100
    # 子文档切块大小（字符数）
    child_chunk_size: int = 200
    child_overlap: int = 50
    # 父文档持久化路径
    document_store_path: Optional[str] = "./parent_docs.json"


class RAGPipeline:
    """
    模块化 RAG 流水线

    普通模式:
        pipeline = RAGPipeline(config)
        pipeline.chunk(files=[...])
        pipeline.embed()
        pipeline.store()
        result = pipeline.retrieve("查询问题")
        response = pipeline.generate("查询问题")

    父子文档模式 (chunking_strategy="parent_child"):
        pipeline = RAGPipeline(config)
        pipeline.chunk(files=[...])         # 自动双层切块
        pipeline.embed()                    # 只向量化子文档
        pipeline.store()                    # 子→向量库，父→文档库
        result = pipeline.retrieve("查询")  # 返回父文档
        response = pipeline.generate("查询")
    """

    def __init__(self, config: Optional[RAGPipelineConfig] = None):
        self.config = config or RAGPipelineConfig()
        self._embedder = None
        self._vector_store = None
        self._retriever = None
        self._generator = None
        self._chunks = None          # ChunkingResult (子文档或普通块)
        self._parent_chunks = None   # list[dict] 父文档列表
        self._document_store = None  # InMemoryStore for parent docs
        self._embeddings = None

    def _get_embedder(self):
        if self._embedder is None:
            if self.config.embedder_type == "sentence_transformer":
                from magic_embedder.strategies import SentenceTransformerEmbedder
                self._embedder = SentenceTransformerEmbedder(model_name=self.config.embedder_model)
            elif self.config.embedder_type == "openai":
                from magic_embedder.strategies import OpenAITextEmbedder
                self._embedder = OpenAITextEmbedder(model_name=self.config.embedder_model)
            elif self.config.embedder_type == "clip":
                from magic_embedder.strategies import CLIPImageEmbedder
                self._embedder = CLIPImageEmbedder(model_name=self.config.embedder_model)
            else:
                raise ValueError(f"Unknown embedder type: {self.config.embedder_type}")
        return self._embedder

    def _get_vector_store(self):
        if self._vector_store is None:
            if self.config.vectorstore_type == "chroma":
                from magic_vectorstore import ChromaVectorStore
                self._vector_store = ChromaVectorStore(
                    collection_name=self.config.collection_name,
                    persist_dir=self.config.persist_dir,
                )
            elif self.config.vectorstore_type == "faiss":
                from magic_vectorstore import FAISSVectorStore
                self._vector_store = FAISSVectorStore(
                    dimension=0,  # 自动从第一个 embedding 检测
                    index_type=self.config.faiss_index_type,
                    metric="cosine",
                )
            else:
                raise ValueError(f"Unknown vectorstore type: {self.config.vectorstore_type}")
        return self._vector_store

    def _get_document_store(self):
        """获取或创建父文档存储 (InMemoryStore)"""
        if self._document_store is None:
            from magic_vectorstore.core.inmemory_store import InMemoryStore
            if self.config.document_store_path:
                import os
                if os.path.exists(self.config.document_store_path):
                    self._document_store = InMemoryStore.load(self.config.document_store_path)
                else:
                    self._document_store = InMemoryStore()
            else:
                self._document_store = InMemoryStore()
        return self._document_store

    def _get_child_retriever(self):
        """创建子文档检索器（用于父子文档模式）"""
        embedder = self._get_embedder()
        vector_store = self._get_vector_store()
        if self.config.retriever_type == "similarity":
            from magic_retriever.strategies import SimilarityRetriever
            return SimilarityRetriever(embedder=embedder, vector_store=vector_store)
        elif self.config.retriever_type == "mmr":
            from magic_retriever.strategies import MMRRetriever
            return MMRRetriever(
                embedder=embedder,
                vector_store=vector_store,
                top_k=self.config.top_k,
                fetch_k=self.config.fetch_k,
                lambda_mult=self.config.lambda_mult,
            )
        else:
            raise ValueError(f"Unknown retriever type: {self.config.retriever_type}")

    def _get_retriever(self):
        if self._retriever is None:
            if self.config.chunking_strategy == "parent_child":
                from magic_retriever.strategies import ParentChildRetriever
                child_retriever = self._get_child_retriever()
                doc_store = self._get_document_store()
                self._retriever = ParentChildRetriever(
                    child_retriever=child_retriever,
                    document_store=doc_store,
                    top_k=self.config.top_k,
                )
            else:
                embedder = self._get_embedder()
                vector_store = self._get_vector_store()
                if self.config.retriever_type == "similarity":
                    from magic_retriever.strategies import SimilarityRetriever
                    self._retriever = SimilarityRetriever(
                        embedder=embedder,
                        vector_store=vector_store,
                    )
                elif self.config.retriever_type == "mmr":
                    from magic_retriever.strategies import MMRRetriever
                    self._retriever = MMRRetriever(
                        embedder=embedder,
                        vector_store=vector_store,
                        top_k=self.config.top_k,
                        fetch_k=self.config.fetch_k,
                        lambda_mult=self.config.lambda_mult,
                    )
                else:
                    raise ValueError(f"Unknown retriever type: {self.config.retriever_type}")
        return self._retriever

    def _get_generator(self):
        if self._generator is None:
            if self.config.generator_type == "openai":
                from magic_generator.strategies import OpenAIGenerator
                self._generator = OpenAIGenerator(model=self.config.generator_model)
            elif self.config.generator_type == "anthropic":
                from magic_generator.strategies import AnthropicGenerator
                self._generator = AnthropicGenerator(model=self.config.generator_model)
            else:
                raise ValueError(f"Unknown generator type: {self.config.generator_type}")
        return self._generator

    def chunk(self, files=None, nodes=None) -> ChunkingResult:
        """
        分块阶段：加载文件并切分为块

        Args:
            files: 文件路径列表，支持 MinerU 的 content_list.json 和 .md 文件
            nodes: 或者直接传入预处理的 Node 列表
        """
        from magic_chunker.loaders import MinerUContentListLoader, MinerUMarkdownLoader

        all_nodes = []

        # 从文件加载
        if files:
            for file_path in files:
                if file_path.endswith('content_list.json'):
                    loader = MinerUContentListLoader(file_path)
                    loaded_nodes = loader.load()
                    all_nodes.extend(loaded_nodes)
                elif file_path.endswith('.md'):
                    loader = MinerUMarkdownLoader(file_path)
                    loaded_nodes = loader.load()
                    all_nodes.extend(loaded_nodes)

        # 从传入的 Node 列表加载
        if nodes:
            all_nodes.extend(nodes)

        if not all_nodes:
            from magic_chunker.core import ChunkingResult
            return ChunkingResult(chunks=[], metadata={"total_nodes": 0})

        if self.config.chunking_strategy == "parent_child":
            from magic_chunker.strategies import ParentChildChunker
            chunker = ParentChildChunker(
                parent_chunk_size=self.config.parent_chunk_size,
                parent_overlap=self.config.parent_overlap,
                child_chunk_size=self.config.child_chunk_size,
                child_overlap=self.config.child_overlap,
            )
            result = chunker.chunk(all_nodes)
            self._chunks = result
            # 保存父文档列表
            self._parent_chunks = result.metadata.get("parent_documents", [])
            return result
        else:
            from magic_chunker.strategies import SemanticChunker
            chunker = SemanticChunker()
            result = chunker.chunk(all_nodes)
            self._chunks = result
            self._parent_chunks = None
            return result

    def embed(self, texts: list[str] = None) -> EmbeddingResult:
        """
        向量化阶段：对文本进行嵌入
        """
        if texts is None and self._chunks:
            texts = [chunk.content for chunk in self._chunks.chunks]

        embedder = self._get_embedder()
        self._embeddings = embedder.embed(texts)
        return self._embeddings

    def store(self, texts: list[str] = None, embeddings: list = None):
        """
        存储阶段：
        - 普通模式：将向量存入向量数据库
        - 父子文档模式：子文档→向量库，父文档→文档库
        """
        if texts is None and self._chunks:
            texts = [chunk.content for chunk in self._chunks.chunks]

        if embeddings is None:
            embeddings = self._embeddings.embeddings if self._embeddings else None

        if embeddings is None:
            raise ValueError("No embeddings provided. Call embed() first or provide embeddings.")

        vector_store = self._get_vector_store()

        from magic_vectorstore.core import VectorEntry
        entries = [
            VectorEntry(
                id=chunk.id if self._chunks else f"chunk_{i}",
                embedding=emb,
                text=text,
                metadata=getattr(chunk, 'metadata', {}) if self._chunks else {},
            )
            for i, (emb, text, chunk) in enumerate(zip(embeddings, texts, self._chunks.chunks))
        ]
        vector_store.add(entries)

        if hasattr(vector_store, 'persist') and self.config.persist_dir:
            vector_store.persist(self.config.persist_dir)

        # 父子文档模式：额外存储父文档
        if self.config.chunking_strategy == "parent_child" and self._parent_chunks:
            doc_store = self._get_document_store()
            doc_store.add(self._parent_chunks)
            if self.config.document_store_path:
                doc_store.persist(self.config.document_store_path)

    def retrieve(self, query: str, top_k: int = None) -> RetrievalResult:
        """
        检索阶段：根据查询检索相关块
        - 普通模式：返回向量检索命中的块
        - 父子文档模式：返回父文档（通过子文档的 parent_id 召回）
        """
        if top_k is None:
            top_k = self.config.top_k

        retriever = self._get_retriever()
        return retriever.retrieve(query, top_k=top_k)

    def generate(self, query: str, context: list[str] = None) -> GenerationResult:
        """
        生成阶段：基于检索结果生成回答
        """
        if context is None:
            retrieval_result = self.retrieve(query)
            context = [chunk.content for chunk in retrieval_result.chunks]

        generator = self._get_generator()
        prompt = GenerationPrompt(
            user_prompt=query,
            context=context,
        )
        return generator.generate(prompt)

    def run(self, query: str) -> str:
        """
        完整流水线：检索 + 生成
        """
        result = self.generate(query)
        return result.response

    def stats(self) -> dict:
        """返回流水线状态统计"""
        stats = {
            "chunking_strategy": self.config.chunking_strategy,
            "total_chunks": len(self._chunks.chunks) if self._chunks else 0,
            "total_parent_docs": len(self._parent_chunks) if self._parent_chunks else 0,
        }
        if self._document_store:
            stats["parent_docs_in_store"] = self._document_store.count()
        return stats
