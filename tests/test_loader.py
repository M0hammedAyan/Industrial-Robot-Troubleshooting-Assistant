"""Unit tests for document loading."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.document_loader import load_document, load_txt


def test_load_txt(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text(
        "SECTION 1 COOLING SYSTEM\n\nClean the filter monthly.\n", encoding="utf-8"
    )
    pages = load_txt(path)
    assert len(pages) == 1
    assert pages[0].source == "sample.txt"
    assert pages[0].page == 1
    assert "filter" in pages[0].text.lower()


def test_unsupported_extension(tmp_path: Path):
    path = tmp_path / "notes.csv"
    path.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_document(path)


def test_empty_txt(tmp_path: Path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Empty"):
        load_txt(path)


def test_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "missing.pdf")
