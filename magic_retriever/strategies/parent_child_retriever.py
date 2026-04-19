"""Parent-Child Retriever: retrieves child chunks then fetches corresponding parent documents."""

from magic_retriever.core import RetrievedChunk, RetrievalResult, BaseRetriever
from magic_vectorstore.core.inmemory_store import InMemoryStore


class ParentChildRetriever(BaseRetriever):
    """
    父子文档检索器。

    1. 用子文档的向量检索 top_k 个相关子块
    2. 根据子块的 parent_id 从 DocumentStore 中拉取完整父文档
    3. 去重后返回父文档列表

    Attributes:
        child_retriever: 子文档检索器（SimilarityRetriever 或 MMRRetriever）
        document_store: 父文档存储 (InMemoryStore)
        top_k: 最终返回的父文档数量
    """

    def __init__(
        self,
        child_retriever: BaseRetriever,
        document_store: InMemoryStore,
        top_k: int = 5,
    ):
        self.child_retriever = child_retriever
        self.document_store = document_store
        self.top_k = top_k

    @property
    def name(self) -> str:
        return f"parent_child_{self.child_retriever.name}"

    @property
    def description(self) -> str:
        return (
            f"Parent-Child Retriever wrapping {self.child_retriever.description}, "
            f"returning {self.top_k} parent documents"
        )

    def retrieve(self, query: str, top_k: int = None) -> RetrievalResult:
        """
        检索父文档。

        Args:
            query: 查询文本
            top_k: 返回的父文档数量（默认 self.top_k）

        Returns:
            RetrievalResult: 包含去重后的父文档内容
        """
        if top_k is None:
            top_k = self.top_k

        # Step 1: 检索子文档（可以多取一些，防止去重后不足）
        child_result = self.child_retriever.retrieve(query, top_k=top_k * 3)

        if not child_result.chunks:
            return RetrievalResult(chunks=[], scores=[], query=query)

        # Step 2: 收集所有 parent_id
        parent_ids: list[str] = []
        child_to_score: dict[str, float] = {}
        for chunk in child_result.chunks:
            pid = chunk.metadata.get("parent_id")
            if pid:
                parent_ids.append(pid)
                if pid not in child_to_score:
                    child_to_score[pid] = chunk.score
                else:
                    child_to_score[pid] = max(child_to_score[pid], chunk.score)

        # Step 3: 去重（同一个父文档可能命中多个子文档）
        unique_parent_ids = list(dict.fromkeys(parent_ids))

        # Step 4: 从 DocumentStore 拉取父文档
        parent_docs: list[RetrievedChunk] = []
        for pid in unique_parent_ids:
            doc = self.document_store.get(pid)
            if doc:
                retrieved = RetrievedChunk(
                    chunk_id=pid,
                    content=doc["content"],
                    score=child_to_score.get(pid, 0.0),
                    metadata=doc.get("metadata", {}),
                    child_chunk_ids=[
                        c.chunk_id for c in child_result.chunks
                        if c.metadata.get("parent_id") == pid
                    ],
                )
                parent_docs.append(retrieved)
                if len(parent_docs) >= top_k:
                    break

        # Step 5: 按分数排序
        parent_docs.sort(key=lambda x: x.score, reverse=True)

        return RetrievalResult(
            chunks=parent_docs,
            query=query,
        )
