"""FAISS vector store implementation."""

import os
from pathlib import Path
from typing import Any

from magic_vectorstore.core import BaseVectorStore, QueryResult, VectorEntry, VectorStoreStats, VectorStoreType

try:
    import numpy as np
except ImportError:
    raise ImportError("numpy is required for FAISSVectorStore. Install with: pip install numpy")

try:
    import faiss
except ImportError:
    raise ImportError("faiss package is required for FAISSVectorStore. Install with: pip install faiss-cpu (or faiss-gpu)")


class FAISSVectorStore(BaseVectorStore):
    """FAISS vector store implementation.

    Wrapper around FAISS for efficient similarity search.

    Attributes:
        dimension: Dimension of the embedding vectors.
        index_type: FAISS index type ('flat', 'ivf', 'hnsw').
        metric: Distance metric ('cosine', 'l2').
        nlist: Number of clusters for IVF index.
        nprobe: Number of clusters to search for IVF index.
    """

    def __init__(
        self,
        dimension: int = 0,  # 0 means auto-detect from first entry
        index_type: str = "flat",
        metric: str = "cosine",
        nlist: int = 100,
        nprobe: int = 10,
    ):
        self.dimension = dimension  # May be updated on first add()
        self.index_type = index_type.lower()
        self.metric = metric.lower()
        self.nlist = nlist
        self.nprobe = nprobe

        self._index = None  # Created on first add() when dimension is known
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}
        self._texts: dict[str, str] = {}
        self._metadatas: dict[str, dict] = {}
        self._next_idx = 0
        self._is_trained = self.index_type == "flat"
        self._dimension_auto_set = dimension == 0

    def _create_index(self):
        """Create FAISS index based on type."""
        if self.metric == "cosine" or self.metric == "ip":
            # Inner product for cosine similarity (with normalized vectors)
            if self.index_type == "flat":
                return faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(self.dimension, 32)
                return index
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(self.dimension)
                return faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            # L2 distance
            if self.index_type == "flat":
                return faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "hnsw":
                return faiss.IndexHNSWFlat(self.dimension, 32)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(self.dimension)
                return faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)

        raise ValueError(f"Unsupported index type or metric: {self.index_type}, {self.metric}")

    @property
    def name(self) -> str:
        return f"faiss_{self.index_type}"

    @property
    def description(self) -> str:
        return f"FAISS vector store ({self.index_type}, {self.metric})"

    def _normalize(self, vectors):
        """Normalize vectors to unit length."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def add(self, entries: list[VectorEntry]) -> None:
        """Add entries to the store.

        Args:
            entries: List of VectorEntry objects to add.
        """
        if not entries:
            return

        # Auto-detect dimension from first entry if needed
        if self._dimension_auto_set and self.dimension == 0:
            self.dimension = len(entries[0].embedding)
            self._index = self._create_index()

        # Validate dimensions
        for entry in entries:
            if len(entry.embedding) != self.dimension:
                raise ValueError(
                    f"Embedding dimension {len(entry.embedding)} does not match "
                    f"store dimension {self.dimension}"
                )

        embeddings = np.array([e.embedding for e in entries], dtype=np.float32)

        # Normalize for cosine similarity
        if self.metric == "cosine":
            embeddings = self._normalize(embeddings)

        # Add to index
        self._index.add(embeddings)

        # Track mappings
        for entry in entries:
            self._id_to_idx[entry.id] = self._next_idx
            self._idx_to_id[self._next_idx] = entry.id
            self._texts[entry.id] = entry.text or ""
            self._metadatas[entry.id] = entry.metadata
            self._next_idx += 1

        # Train index if needed (for IVF)
        if not self._is_trained and self._index.is_trained:
            self._is_trained = True

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Search for similar vectors.

        Args:
            query_vector: The query vector.
            top_k: Number of results to return.
            filter_metadata: Optional metadata filter (not supported in FAISS basic search).

        Returns:
            QueryResult with matching entries and scores.
        """
        query = np.array([query_vector], dtype=np.float32)

        # Normalize for cosine similarity
        if self.metric == "cosine":
            query = self._normalize(query)

        # Set nprobe for IVF
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = self.nprobe

        # Search
        distances, indices = self._index.search(query, top_k)

        entries: list[VectorEntry] = []
        scores: list[float] = []

        for i, idx in enumerate(indices[0]):
            if idx < 0:
                continue

            entry_id = self._idx_to_id.get(int(idx), f"unknown_{idx}")
            dist = distances[0][i]

            # Convert distance to similarity score
            if self.metric == "cosine" or self.metric == "ip":
                score = float(dist)
            else:
                # L2 distance - convert to similarity
                score = 1.0 / (1.0 + float(dist))

            entry = VectorEntry(
                id=entry_id,
                embedding=[],
                text=self._texts.get(entry_id, ""),
                metadata=self._metadatas.get(entry_id, {}),
            )
            entries.append(entry)
            scores.append(score)

        return QueryResult(
            entries=entries,
            scores=scores,
            query=query_vector,
        )

    def delete(self, entry_ids: list[str]) -> None:
        """Delete entries from the store.

        Note: FAISS doesn't support direct deletion. This is a no-op for safety.
        Consider rebuilding the index if deletion is needed.

        Args:
            entry_ids: List of entry IDs to delete.
        """
        # FAISS IndexFlatIP/L2 doesn't support deletion
        # For production, consider using a workaround or a different index type
        pass

    def persist(self, path: str) -> None:
        """Persist the index to disk.

        Args:
            path: Path to save the index (without extension).
        """
        path_prefix = path.replace(".faiss", "").replace(".idx", "")
        index_path = f"{path_prefix}.faiss"

        faiss.write_index(self._index, index_path)

        # Also save the metadata
        import json

        metadata_path = f"{path_prefix}_meta.json"
        metadata = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "id_to_idx": self._id_to_idx,
            "idx_to_id": {str(k): v for k, v in self._idx_to_id.items()},
            "texts": self._texts,
            "metadatas": self._metadatas,
            "next_idx": self._next_idx,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "FAISSVectorStore":
        """Load a persisted FAISS index.

        Args:
            path: Path to the index (without extension).

        Returns:
            Loaded FAISSVectorStore instance.
        """
        path_prefix = path.replace(".faiss", "").replace(".idx", "")
        index_path = f"{path_prefix}.faiss"
        metadata_path = f"{path_prefix}_meta.json"

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Create instance
        store = cls(
            dimension=metadata["dimension"],
            index_type=metadata["index_type"],
            metric=metadata["metric"],
            nlist=metadata.get("nlist", 100),
            nprobe=metadata.get("nprobe", 10),
        )

        # Load index
        store._index = faiss.read_index(index_path)
        store._id_to_idx = metadata["id_to_idx"]
        store._idx_to_id = {int(k): v for k, v in metadata["idx_to_id"].items()}
        store._texts = metadata["texts"]
        store._metadatas = metadata["metadatas"]
        store._next_idx = metadata["next_idx"]
        store._is_trained = True

        return store

    def stats(self) -> VectorStoreStats:
        """Get statistics about the store."""
        return VectorStoreStats(
            total_entries=self._index.ntotal,
            dimension=self.dimension,
            store_type=VectorStoreType.FAISS,
            metadata={
                "index_type": self.index_type,
                "metric": self.metric,
                "is_trained": self._is_trained,
            },
        )
