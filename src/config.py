"""Central configuration for the Industrial Robot Troubleshooting Assistant."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root: industrial_robot_assistant/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.example")


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- Model ---
MODEL_NAME = _env("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
# Scale up when download/VRAM allow: Qwen/Qwen2.5-1.5B-Instruct or Qwen/Qwen2.5-3B-Instruct
FALLBACK_MODEL_NAME = _env("FALLBACK_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEVICE = _env("DEVICE", "auto")  # auto | cuda | cpu
LOAD_IN_4BIT = _env_bool("LOAD_IN_4BIT", True)
MAX_NEW_TOKENS = _env_int("MAX_NEW_TOKENS", 512)
TEMPERATURE = _env_float("TEMPERATURE", 0.2)
TOP_P = _env_float("TOP_P", 0.9)

# --- Retrieval / chunking ---
TOP_K = _env_int("TOP_K", 5)
CHUNK_SIZE = _env_int("CHUNK_SIZE", 600)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 120)
# Cosine similarity threshold for FAISS IndexFlatIP on L2-normalized embeddings.
# Calibrated on the live 3513-chunk index: unrelated queries scored ~0.29;
# topical-but-unsupported hits (e.g. hydraulic pressure) can still score ~0.40+,
# so evidence validation is also required beyond this gate.
RETRIEVAL_THRESHOLD = _env_float("RETRIEVAL_THRESHOLD", 0.35)
# Back-compat alias
MIN_RELEVANCE_SCORE = _env_float("MIN_RELEVANCE_SCORE", RETRIEVAL_THRESHOLD)

# --- Paths ---
DOCUMENT_PATH = Path(_env("DOCUMENT_PATH", str(PROJECT_ROOT / "data" / "manuals")))
if not DOCUMENT_PATH.is_absolute():
    DOCUMENT_PATH = PROJECT_ROOT / DOCUMENT_PATH

VECTOR_STORE_PATH = Path(_env("VECTOR_STORE_PATH", str(PROJECT_ROOT / "vector_store")))
if not VECTOR_STORE_PATH.is_absolute():
    VECTOR_STORE_PATH = PROJECT_ROOT / VECTOR_STORE_PATH

LOG_DIR = Path(_env("LOG_DIR", str(PROJECT_ROOT / "logs")))
if not LOG_DIR.is_absolute():
    LOG_DIR = PROJECT_ROOT / LOG_DIR

FAISS_INDEX_FILE = VECTOR_STORE_PATH / "index.faiss"
FAISS_META_FILE = VECTOR_STORE_PATH / "chunks_meta.json"
INDEX_MANIFEST_FILE = VECTOR_STORE_PATH / "manifest.json"

# --- App ---
APP_NAME = "Industrial Robot Troubleshooting Assistant"
APP_SUBTITLE = "AI-powered document-based troubleshooting assistant"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def ensure_directories() -> None:
    """Create required directories if they do not exist."""
    DOCUMENT_PATH.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def summary() -> dict:
    """Return a JSON-serializable summary of active settings."""
    return {
        "model_name": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "device": DEVICE,
        "load_in_4bit": LOAD_IN_4BIT,
        "top_k": TOP_K,
        "retrieval_threshold": RETRIEVAL_THRESHOLD,
        "score_type": "cosine_similarity (IndexFlatIP + L2-normalized embeddings)",
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "document_path": str(DOCUMENT_PATH),
        "vector_store_path": str(VECTOR_STORE_PATH),
    }
