"""Sentence-Transformers embedding wrapper with simple caching."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

from src.config import DEVICE, EMBEDDING_MODEL
from src.logging_utils import get_logger

logger = get_logger(__name__)

_model = None


def _resolve_device(device: Optional[str] = None) -> str:
    device = (device or DEVICE or "auto").lower()
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def get_embedding_model(model_name: Optional[str] = None, device: Optional[str] = None):
    """Load and cache the SentenceTransformer embedding model."""
    global _model
    name = model_name or EMBEDDING_MODEL
    resolved = _resolve_device(device)
    if _model is not None:
        return _model
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s (device=%s)", name, resolved)
    _model = SentenceTransformer(name, device=resolved)
    logger.info("Embedding model ready. dim=%s", _model.get_embedding_dimension())
    return _model


def embed_texts(
    texts: Sequence[str],
    model_name: Optional[str] = None,
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """Embed a list of texts; returns float32 L2-normalized vectors (N, D)."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = get_embedding_model(model_name=model_name)
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(query: str, model_name: Optional[str] = None) -> np.ndarray:
    """Embed a single query string; returns shape (D,)."""
    vecs = embed_texts([query], model_name=model_name)
    return vecs[0]


def embedding_dimension(model_name: Optional[str] = None) -> int:
    model = get_embedding_model(model_name=model_name)
    return int(model.get_embedding_dimension())
