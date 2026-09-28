"""Retrieval tests (require built index or build a tiny in-memory store)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunker import Chunk
from src.embeddings import embed_texts
from src.retriever import Retriever, extract_error_codes
from src.vector_store import VectorStore


def test_extract_error_codes():
    codes = extract_error_codes("What does error E-123 mean and alarm 410?")
    assert any("E-123" in c or "E123" in c for c in codes)
    assert any("410" in c for c in codes)


def test_faiss_roundtrip(tmp_path: Path):
    pytest.importorskip("faiss")
    pytest.importorskip("sentence_transformers")

    chunks = [
        Chunk(
            text="Alarm A-305 means controller overtemperature. Clean filters and check fans.",
            source="troubleshooting_manual.pdf",
            page=18,
            section="ALARM CODE REFERENCE",
            chunk_id="c1",
        ),
        Chunk(
            text="Battery replacement interval is every 24 months for encoder backup.",
            source="maintenance_manual.pdf",
            page=42,
            section="BATTERY AND MEMORY",
            chunk_id="c2",
        ),
        Chunk(
            text="Never bypass an emergency stop device on the industrial robot cell.",
            source="safety_manual.pdf",
            page=3,
            section="EMERGENCY STOP",
            chunk_id="c3",
        ),
    ]
    embeddings = embed_texts([c.text for c in chunks])
    store = VectorStore(tmp_path)
    store.build(embeddings, chunks)
    store.save(document_names=["troubleshooting_manual.pdf"])

    store2 = VectorStore(tmp_path)
    store2.load()
    assert store2.size == 3

    retriever = Retriever(store2)
    hits = retriever.retrieve("What does alarm A-305 mean?", top_k=2, min_score=0.1)
    assert hits
    assert "305" in hits[0]["text"] or hits[0]["source"] == "troubleshooting_manual.pdf"
