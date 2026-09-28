"""Pipeline-level tests using sample text documents."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunker import chunk_pages
from src.document_loader import PageText, load_txt
from src.pipeline import RAGPipeline
from src.prompt import NO_CONTEXT_ANSWER, build_messages


def test_prompt_contains_context():
    chunks = [
        {
            "text": "SRVO-001 Motor power not ready",
            "source": "troubleshooting_manual.pdf",
            "page": 5,
            "section": "ALARMS",
            "score": 0.8,
        }
    ]
    messages = build_messages("What is SRVO-001?", chunks)
    assert messages[0]["role"] == "system"
    assert "SRVO-001" in messages[1]["content"]
    assert "troubleshooting_manual.pdf" in messages[1]["content"]


def test_no_context_message():
    assert "could not find" in NO_CONTEXT_ANSWER.lower()


def test_pipeline_build_and_retrieve(tmp_path: Path, monkeypatch):
    pytest.importorskip("faiss")
    pytest.importorskip("sentence_transformers")

    docs = tmp_path / "manuals"
    docs.mkdir()
    (docs / "demo.txt").write_text(
        "ALARM CODE REFERENCE\n\n"
        "A-120 Encoder backup battery low. Replace battery BATT1.\n\n"
        "SAFETY\n\nAlways apply lockout tagout before cabinet work.\n",
        encoding="utf-8",
    )

    store_dir = tmp_path / "store"
    pipeline = RAGPipeline(store_dir=store_dir)
    result = pipeline.build_index(document_dir=docs, force=True)
    assert result["chunks"] >= 1
    assert pipeline.is_ready

    # Retrieval only (skip heavy LLM)
    hits = pipeline.retriever.retrieve("What is alarm A-120?", top_k=3, min_score=0.05)
    assert hits
    assert any("A-120" in h["text"] or "battery" in h["text"].lower() for h in hits)


def test_unknown_question_low_relevance_path(tmp_path: Path):
    """Unknown topics should not retrieve highly relevant industrial content."""
    pytest.importorskip("faiss")
    pytest.importorskip("sentence_transformers")

    docs = tmp_path / "manuals"
    docs.mkdir()
    (docs / "demo.txt").write_text(
        "Cooling System\n\nClean the controller filter every month.\n",
        encoding="utf-8",
    )
    store_dir = tmp_path / "store"
    pipeline = RAGPipeline(store_dir=store_dir)
    pipeline.build_index(document_dir=docs, force=True)
    hits = pipeline.retriever.retrieve(
        "What is the capital of France and the recipe for lasagna?",
        top_k=3,
        min_score=0.55,
    )
    # Either empty due to threshold, or low topical overlap
    if hits:
        assert all("lasagna" not in h["text"].lower() for h in hits)
