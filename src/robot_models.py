"""Robot model detection from queries and document filenames."""

from __future__ import annotations

import re
from typing import List, Optional

# Longer / more specific tokens first to avoid partial clashes.
KNOWN_MODELS = [
    "UR16e",
    "UR12e",
    "UR10e",
    "UR3e",
    "UR5e",
    "UR7e",
    "UR8 Long",
    "UR8",
    "UR18",
    "UR20",
    "UR30",
]

# Filename / free-text patterns (underscores treated as spaces before match).
_MODEL_PATTERNS = [
    (re.compile(r"\bUR16e\b", re.I), "UR16e"),
    (re.compile(r"\bUR12e\b", re.I), "UR12e"),
    (re.compile(r"\bUR10e\b", re.I), "UR10e"),
    (re.compile(r"\bUR3e\b", re.I), "UR3e"),
    (re.compile(r"\bUR5e\b", re.I), "UR5e"),
    (re.compile(r"\bUR7e\b", re.I), "UR7e"),
    (re.compile(r"\bUR8[\s_-]*Long\b", re.I), "UR8 Long"),
    (re.compile(r"\bUR8\b", re.I), "UR8"),
    (re.compile(r"\bUR18\b", re.I), "UR18"),
    (re.compile(r"\bUR20\b", re.I), "UR20"),
    (re.compile(r"\bUR30\b", re.I), "UR30"),
]


def normalize_source_text(value: str) -> str:
    return (value or "").replace("_", " ").replace("-", " ")


def detect_models_in_text(text: str) -> List[str]:
    """Return unique robot models mentioned in text (query or filename)."""
    hay = normalize_source_text(text)
    found: List[str] = []
    for pattern, label in _MODEL_PATTERNS:
        if pattern.search(hay) and label not in found:
            found.append(label)
    return found


def detect_query_model(question: str) -> Optional[str]:
    """Primary model mentioned in the user question, if any."""
    models = detect_models_in_text(question or "")
    return models[0] if models else None


def infer_model_from_source(source: str) -> Optional[str]:
    """Best-effort model label from a document filename."""
    models = detect_models_in_text(source or "")
    if not models:
        return None
    # Prefer the most specific e-Series style match when several appear.
    return models[0]


def chunk_matches_model(chunk: dict, model: str) -> bool:
    """True if chunk source/text/model metadata refers to the given model."""
    if not model:
        return True
    explicit = (chunk.get("model") or "").strip()
    if explicit:
        return explicit.lower() == model.lower()

    source = chunk.get("source") or ""
    text = chunk.get("text") or ""
    inferred = infer_model_from_source(source)
    if inferred and inferred.lower() == model.lower():
        return True
    # Generic shared manuals (ErrorCodes, e-Series) — allow as secondary evidence
    # only when the text itself mentions the model.
    models_in_text = detect_models_in_text(text)
    if model in models_in_text:
        return True
    return False


def is_shared_manual(source: str) -> bool:
    """Documents that intentionally cover multiple / all models."""
    name = (source or "").lower()
    shared_tokens = (
        "errorcodes",
        "error codes",
        "e-series",
        "e_series",
        "service_manual",
        "service manual",
        "calibration",
        "safety",
        "user_manual_en_global.pdf",  # generic untitled UR manual
    )
    return any(tok in name for tok in shared_tokens)
