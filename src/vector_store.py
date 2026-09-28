"""FAISS vector store with metadata persistence."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.chunker import Chunk
from src.config import (
    FAISS_INDEX_FILE,
    FAISS_META_FILE,
    INDEX_MANIFEST_FILE,
    VECTOR_STORE_PATH,
    ensure_directories,
)
from src.logging_utils import get_logger

logger = get_logger(__name__)


class VectorStore:
    """FAISS IndexFlatIP store backed by JSON metadata."""

    def __init__(self, store_dir: Optional[Path] = None) -> None:
        ensure_directories()
        self.store_dir = Path(store_dir) if store_dir else VECTOR_STORE_PATH
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_dir / "index.faiss"
        self.meta_path = self.store_dir / "chunks_meta.json"
        self.manifest_path = self.store_dir / "manifest.json"
        self.index = None
        self.chunks: List[dict] = []
        self.dimension: Optional[int] = None

    @property
    def size(self) -> int:
        if self.index is None:
            return 0
        return int(self.index.ntotal)

    def exists(self) -> bool:
        return self.index_path.exists() and self.meta_path.exists()

    def build(self, embeddings: np.ndarray, chunks: Sequence[Chunk | dict]) -> None:
        """Create a new FAISS index from embeddings and chunk metadata."""
        import faiss

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2-D (N, D)")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings length mismatch")
        if embeddings.shape[0] == 0:
            raise ValueError("Cannot build index from empty embeddings")

        vectors = np.asarray(embeddings, dtype=np.float32)
        # Assume caller already L2-normalized; re-normalize for safety
        faiss.normalize_L2(vectors)

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        meta: List[dict] = []
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, Chunk):
                item = chunk.to_dict()
            else:
                item = dict(chunk)
            item["faiss_id"] = i
            meta.append(item)

        self.index = index
        self.chunks = meta
        self.dimension = dim
        logger.info("Built FAISS index — vectors=%d dim=%d", self.size, dim)

    def add(self, embeddings: np.ndarray, chunks: Sequence[Chunk | dict]) -> None:
        """Append vectors/chunks to an existing index."""
        import faiss

        if self.index is None:
            self.build(embeddings, chunks)
            return

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.shape[0] == 0:
            return
        if vectors.shape[1] != self.dimension:
            raise ValueError("Embedding dimension mismatch with existing index")
        faiss.normalize_L2(vectors)
        start = self.size
        self.index.add(vectors)
        for i, chunk in enumerate(chunks):
            item = chunk.to_dict() if isinstance(chunk, Chunk) else dict(chunk)
            item["faiss_id"] = start + i
            self.chunks.append(item)
        logger.info("Added %d vectors — total=%d", vectors.shape[0], self.size)

    def save(self, document_names: Optional[List[str]] = None) -> None:
        """Persist index, metadata, and a simple manifest."""
        import faiss

        if self.index is None:
            raise RuntimeError("No index to save")
        self.store_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_vectors": self.size,
            "dimension": self.dimension,
            "documents": document_names
            or sorted({c.get("source", "") for c in self.chunks}),
        }
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Saved FAISS index to %s (%d vectors)", self.store_dir, self.size)

    def load(self) -> None:
        """Load index and metadata from disk."""
        import faiss

        if not self.exists():
            raise FileNotFoundError(
                f"FAISS index not found in {self.store_dir}. Build the index first."
            )
        try:
            self.index = faiss.read_index(str(self.index_path))
            self.chunks = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.dimension = int(self.index.d)
        except Exception as exc:
            logger.error("Corrupt or unreadable FAISS index: %s", exc)
            raise RuntimeError(f"Corrupt FAISS index in {self.store_dir}") from exc
        logger.info(
            "Loaded FAISS index — vectors=%d dim=%d", self.size, self.dimension
        )

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple[dict, float]]:
        """Return top_k (chunk_meta, score) pairs. Score is cosine similarity."""
        if self.index is None:
            raise RuntimeError("Index not loaded")
        if self.size == 0:
            return []

        import faiss

        vec = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        k = min(top_k, self.size)
        scores, indices = self.index.search(vec, k)
        results: List[Tuple[dict, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.chunks[int(idx)]
            results.append((meta, float(score)))
        return results

    def clear(self) -> None:
        """Remove persisted index files and reset in-memory state."""
        self.index = None
        self.chunks = []
        self.dimension = None
        for path in (self.index_path, self.meta_path, self.manifest_path):
            if path.exists():
                path.unlink()
        logger.info("Cleared vector store at %s", self.store_dir)

    def document_count(self) -> int:
        return len({c.get("source") for c in self.chunks if c.get("source")})

    def get_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {
            "num_vectors": self.size,
            "dimension": self.dimension,
            "documents": sorted({c.get("source", "") for c in self.chunks}),
        }
