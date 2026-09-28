"""Unit tests for chunking."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunker import chunk_page, chunk_pages
from src.document_loader import PageText


def test_chunk_preserves_metadata():
    page = PageText(
        text=("word " * 700).strip(),
        source="maintenance_manual.pdf",
        page=42,
        section="Cooling System",
    )
    chunks = chunk_page(page, chunk_size=200, chunk_overlap=40)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.source == "maintenance_manual.pdf"
        assert c.page == 42
        assert c.section == "Cooling System"
        assert c.text


def test_overlap_exists():
    words = [f"w{i}" for i in range(350)]
    page = PageText(text=" ".join(words), source="a.pdf", page=1, section="S")
    chunks = chunk_page(page, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 3
    # Last words of first chunk should appear near start of second when overlap works
    first_tail = chunks[0].text.split()[-15:]
    second_head = set(chunks[1].text.split()[:40])
    assert any(w in second_head for w in first_tail)


def test_chunk_pages_count():
    pages = [
        PageText(text=("alpha " * 250).strip(), source="a.pdf", page=1),
        PageText(text=("beta " * 250).strip(), source="a.pdf", page=2),
    ]
    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 4
