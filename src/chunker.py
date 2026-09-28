"""Configurable text chunking that preserves source/page/section metadata."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional, Sequence

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.document_loader import PageText
from src.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A retrieval unit with provenance metadata."""

    text: str
    source: str
    page: int
    section: str = ""
    chunk_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _tokenize_words(text: str) -> List[str]:
    return text.split()


def _join_words(words: Sequence[str]) -> str:
    return " ".join(words).strip()


def _split_into_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_page(
    page: PageText,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    start_index: int = 0,
) -> List[Chunk]:
    """
    Split a page into overlapping word-based chunks.

    Prefers paragraph boundaries when a paragraph fits inside chunk_size.
    """
    if chunk_size < 50:
        raise ValueError("chunk_size must be >= 50")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    paragraphs = _split_into_paragraphs(page.text)
    if not paragraphs:
        return []

    chunks: List[Chunk] = []
    buffer_words: List[str] = []
    idx = start_index

    def flush(force: bool = False) -> None:
        nonlocal idx, buffer_words
        if not buffer_words:
            return
        if len(buffer_words) < max(40, chunk_size // 5) and not force and chunks:
            # Keep very small leftovers with previous content via overlap handling
            return
        text = _join_words(buffer_words)
        if not text:
            return
        chunk_id = f"{page.source}::p{page.page}::c{idx}"
        chunks.append(
            Chunk(
                text=text,
                source=page.source,
                page=page.page,
                section=page.section or "",
                chunk_id=chunk_id,
            )
        )
        idx += 1
        if chunk_overlap > 0 and len(buffer_words) > chunk_overlap:
            buffer_words = buffer_words[-chunk_overlap:]
        else:
            buffer_words = []

    for para in paragraphs:
        words = _tokenize_words(para)
        # Oversized paragraph: hard-split by words
        if len(words) > chunk_size:
            flush(force=True)
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                piece = words[start:end]
                chunk_id = f"{page.source}::p{page.page}::c{idx}"
                chunks.append(
                    Chunk(
                        text=_join_words(piece),
                        source=page.source,
                        page=page.page,
                        section=page.section or "",
                        chunk_id=chunk_id,
                    )
                )
                idx += 1
                if end >= len(words):
                    buffer_words = piece[-chunk_overlap:] if chunk_overlap else []
                    break
                start = max(0, end - chunk_overlap)
            continue

        if len(buffer_words) + len(words) <= chunk_size:
            buffer_words.extend(words)
        else:
            flush(force=True)
            buffer_words.extend(words)
            if len(buffer_words) >= chunk_size:
                flush(force=True)

    flush(force=True)
    return chunks


def chunk_pages(
    pages: Sequence[PageText],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Chunk]:
    """Chunk a sequence of pages into metadata-preserving chunks."""
    chunk_size = CHUNK_SIZE if chunk_size is None else chunk_size
    chunk_overlap = CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap

    all_chunks: List[Chunk] = []
    for page in pages:
        page_chunks = chunk_page(
            page,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            start_index=len(all_chunks),
        )
        all_chunks.extend(page_chunks)

    logger.info(
        "Chunking complete — pages=%d chunks=%d (size=%d overlap=%d)",
        len(pages),
        len(all_chunks),
        chunk_size,
        chunk_overlap,
    )
    return all_chunks


def chunks_to_dicts(chunks: Iterable[Chunk]) -> List[dict]:
    return [c.to_dict() for c in chunks]
