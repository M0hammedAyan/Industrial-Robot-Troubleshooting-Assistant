"""PDF / text document loading with page and section metadata."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from src.config import DOCUMENT_PATH, SUPPORTED_EXTENSIONS
from src.logging_utils import get_logger

logger = get_logger(__name__)

HEADING_RE = re.compile(
    r"^(?:"
    r"[A-Z][A-Z0-9][A-Z0-9\s\-:/]{2,80}"  # ALL CAPS heading
    r"|"
    r"(?:\d+(?:\.\d+)*)\s+[A-Z][^\n]{2,80}"  # numbered section
    r"|"
    r"(?:Chapter|Section|CHAPTER|SECTION)\s+\d+[^\n]{0,60}"
    r")$"
)


@dataclass
class PageText:
    """Text extracted from a single document page."""

    text: str
    source: str
    page: int
    section: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _clean_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _guess_section(page_text: str, previous_section: str = "") -> str:
    """Best-effort section heading from the first heading-like line on a page."""
    for line in page_text.splitlines()[:12]:
        candidate = line.strip()
        if len(candidate) < 4 or len(candidate) > 100:
            continue
        if HEADING_RE.match(candidate):
            return candidate
    return previous_section


def load_pdf(path: Path) -> List[PageText]:
    """Extract text from a PDF, preserving page numbers."""
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz  # PyMuPDF legacy import
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is required for PDF loading. Install with: pip install pymupdf"
            ) from exc

    pages: List[PageText] = []
    source = path.name
    try:
        doc = fitz.open(path)
    except Exception as exc:
        logger.error("Invalid or unreadable PDF: %s (%s)", path, exc)
        raise ValueError(f"Invalid PDF: {path.name}") from exc

    total_pages = doc.page_count
    if total_pages == 0:
        doc.close()
        raise ValueError(f"Empty PDF: {path.name}")

    section = ""
    for i in range(total_pages):
        page = doc.load_page(i)
        raw = page.get_text("text") or ""
        cleaned = _clean_whitespace(raw)
        if not cleaned:
            continue
        section = _guess_section(cleaned, section)
        pages.append(
            PageText(text=cleaned, source=source, page=i + 1, section=section)
        )

    doc.close()
    logger.info(
        "Loaded PDF %s — pages_with_text=%d / total=%d",
        source,
        len(pages),
        total_pages,
    )
    if not pages:
        raise ValueError(f"No extractable text in PDF: {path.name}")
    return pages


def load_txt(path: Path) -> List[PageText]:
    """Load a plain-text file as a single logical page (page=1)."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    cleaned = _clean_whitespace(text)
    if not cleaned:
        raise ValueError(f"Empty text document: {path.name}")
    section = _guess_section(cleaned, "")
    logger.info("Loaded TXT %s — characters=%d", path.name, len(cleaned))
    return [PageText(text=cleaned, source=path.name, page=1, section=section)]


def load_docx(path: Path) -> List[PageText]:
    """Load a DOCX file if python-docx is available; otherwise raise a clear error."""
    try:
        import docx  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "python-docx is required for DOCX files. Install with: pip install python-docx"
        ) from exc

    document = docx.Document(str(path))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    cleaned = _clean_whitespace("\n".join(paragraphs))
    if not cleaned:
        raise ValueError(f"Empty DOCX document: {path.name}")
    section = _guess_section(cleaned, "")
    logger.info("Loaded DOCX %s — characters=%d", path.name, len(cleaned))
    return [PageText(text=cleaned, source=path.name, page=1, section=section)]


def load_document(path: Path | str) -> List[PageText]:
    """Load a single supported document."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".txt":
        return load_txt(path)
    if suffix == ".docx":
        return load_docx(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def list_documents(directory: Optional[Path] = None) -> List[Path]:
    """List supported documents in the manuals directory."""
    directory = Path(directory) if directory else DOCUMENT_PATH
    if not directory.exists():
        logger.warning("Document directory does not exist: %s", directory)
        return []
    files = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return files


def load_all_documents(directory: Optional[Path] = None) -> List[PageText]:
    """Load all supported documents from a directory."""
    directory = Path(directory) if directory else DOCUMENT_PATH
    pages: List[PageText] = []
    files = list_documents(directory)
    logger.info("Found %d document(s) in %s", len(files), directory)
    for path in files:
        try:
            pages.extend(load_document(path))
        except Exception as exc:
            logger.error("Skipping %s: %s", path.name, exc)
    return pages


def pages_to_dicts(pages: Iterable[PageText]) -> List[dict]:
    return [p.to_dict() for p in pages]
