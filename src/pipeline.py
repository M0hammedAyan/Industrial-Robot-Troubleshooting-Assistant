"""End-to-end RAG pipeline: ingest → index → retrieve → generate."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.chunker import Chunk, chunk_pages, chunks_to_dicts
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_PATH,
    RETRIEVAL_THRESHOLD,
    TOP_K,
    VECTOR_STORE_PATH,
    ensure_directories,
)
from src.document_loader import list_documents, load_all_documents, load_document
from src.embeddings import embed_texts
from src.grounding import (
    extract_error_codes,
    extractive_error_answer,
    filter_chunks_for_codes,
    filter_evidence_chunks,
    should_refuse_generation,
)
from src.llm import active_model_name, generate_answer
from src.logging_utils import get_logger
from src.prompt import NO_CONTEXT_ANSWER, build_messages
from src.retriever import Retriever
from src.robot_models import infer_model_from_source
from src.vector_store import VectorStore

logger = get_logger(__name__)


def _file_fingerprint(paths: Sequence[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name.lower()):
        stat = path.stat()
        h.update(path.name.encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(int(stat.st_mtime)).encode("utf-8"))
    return h.hexdigest()


class RAGPipeline:
    """Orchestrates document indexing and question answering."""

    def __init__(self, store_dir: Optional[Path] = None) -> None:
        ensure_directories()
        self.store = VectorStore(store_dir or VECTOR_STORE_PATH)
        self.retriever = Retriever(self.store)
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready and self.store.index is not None and self.store.size > 0

    def load_or_build(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Load an existing index or build one from DOCUMENT_PATH."""
        docs = list_documents(DOCUMENT_PATH)
        fingerprint = _file_fingerprint(docs) if docs else ""

        if (
            not force_rebuild
            and self.store.exists()
        ):
            try:
                self.store.load()
                manifest = self.store.get_manifest()
                saved_fp = manifest.get("fingerprint")
                if saved_fp and fingerprint and saved_fp != fingerprint:
                    logger.warning(
                        "Documents on disk differ from the saved index fingerprint, "
                        "but the existing FAISS index will be kept. Click "
                        "'Rebuild index' in the UI only if you intend to re-index."
                    )
                self._ready = True
                logger.info("Using existing FAISS index (%d chunks)", self.store.size)
                return {
                    "action": "loaded",
                    "chunks": self.store.size,
                    "documents": self.store.document_count(),
                }
            except Exception as exc:
                logger.warning("Could not load index (%s); rebuilding.", exc)

        return self.build_index(force=True)

    def build_index(
        self,
        document_dir: Optional[Path] = None,
        force: bool = False,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Extract, chunk, embed, and persist a fresh FAISS index."""
        t0 = time.perf_counter()
        directory = Path(document_dir) if document_dir else DOCUMENT_PATH
        pages = load_all_documents(directory)
        if not pages:
            raise RuntimeError(
                f"No documents with extractable text found in {directory}"
            )

        chunks = chunk_pages(
            pages,
            chunk_size=chunk_size or CHUNK_SIZE,
            chunk_overlap=chunk_overlap or CHUNK_OVERLAP,
        )
        if not chunks:
            raise RuntimeError("Chunking produced zero chunks")

        logger.info("Generating embeddings for %d chunks...", len(chunks))
        embeddings = embed_texts([c.text for c in chunks], show_progress=True)
        self.store.build(embeddings, chunks)

        docs = list_documents(directory)
        self.store.save(document_names=[p.name for p in docs])
        # Attach fingerprint to manifest
        manifest = self.store.get_manifest()
        manifest["fingerprint"] = _file_fingerprint(docs)
        self.store.manifest_path.write_text(
            __import__("json").dumps(manifest, indent=2), encoding="utf-8"
        )

        self._ready = True
        elapsed = time.perf_counter() - t0
        logger.info(
            "Index build complete — pages=%d chunks=%d docs=%d time=%.2fs",
            len(pages),
            len(chunks),
            len(docs),
            elapsed,
        )
        return {
            "action": "built",
            "pages": len(pages),
            "chunks": len(chunks),
            "documents": len(docs),
            "elapsed_sec": round(elapsed, 2),
        }

    def ingest_file(self, path: Path | str) -> Dict[str, Any]:
        """Ingest a newly uploaded document and update the FAISS index."""
        path = Path(path)
        t0 = time.perf_counter()
        pages = load_document(path)
        chunks = chunk_pages(pages)
        if not chunks:
            raise ValueError(f"No chunks produced from {path.name}")

        embeddings = embed_texts([c.text for c in chunks])

        if self.store.index is None:
            if self.store.exists():
                self.store.load()
            else:
                self.store.build(embeddings, chunks)
                self.store.save()
                self._ready = True
                return {
                    "action": "created",
                    "source": path.name,
                    "chunks": len(chunks),
                    "elapsed_sec": round(time.perf_counter() - t0, 2),
                }

        # Remove existing chunks from same source then rebuild for correctness
        remaining = [c for c in self.store.chunks if c.get("source") != path.name]
        if remaining:
            # Rebuild from remaining + new to avoid orphan vectors
            all_chunk_dicts = remaining + [c.to_dict() for c in chunks]
            all_texts = [c["text"] for c in all_chunk_dicts]
            all_embeddings = embed_texts(all_texts)
            rebuilt = [
                Chunk(
                    text=c["text"],
                    source=c.get("source", ""),
                    page=int(c.get("page", 1)),
                    section=c.get("section", ""),
                    chunk_id=c.get("chunk_id", ""),
                )
                for c in all_chunk_dicts
            ]
            self.store.build(all_embeddings, rebuilt)
        else:
            self.store.add(embeddings, chunks)

        docs = list_documents(DOCUMENT_PATH)
        self.store.save(document_names=[p.name for p in docs])
        manifest = self.store.get_manifest()
        manifest["fingerprint"] = _file_fingerprint(docs)
        self.store.manifest_path.write_text(
            __import__("json").dumps(manifest, indent=2), encoding="utf-8"
        )
        self._ready = True
        elapsed = time.perf_counter() - t0
        logger.info(
            "Ingested %s — chunks=%d total=%d time=%.2fs",
            path.name,
            len(chunks),
            self.store.size,
            elapsed,
        )
        return {
            "action": "updated",
            "source": path.name,
            "chunks_added": len(chunks),
            "total_chunks": self.store.size,
            "elapsed_sec": round(elapsed, 2),
        }

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Retrieve context and generate a grounded answer."""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        if not self.is_ready:
            raise RuntimeError(
                "Knowledge base is not ready. Build or load the index first."
            )

        t0 = time.perf_counter()
        k = TOP_K if top_k is None else top_k
        retrieved = self.retriever.retrieve(question, top_k=k, min_score=min_score)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        if not retrieved:
            answer = NO_CONTEXT_ANSWER
            generation_ms = 0.0
        else:
            # Check for error codes before filtering
            codes = extract_error_codes(question)
            retrieved = filter_chunks_for_codes(retrieved, question)
            # If error codes were asked but no chunks match, return specific message
            if codes and not retrieved:
                answer = (
                    f"I could not find error code {', '.join(codes)} "
                    "in the available documentation."
                )
                generation_ms = 0.0
            else:
                max_score = max((r.get("score", 0.0) for r in retrieved), default=0.0)
                refuse, reason = should_refuse_generation(question, retrieved, max_score, RETRIEVAL_THRESHOLD)
                t1 = time.perf_counter()
                try:
                    if refuse:
                        logger.info("Refusing generation (%s)", reason)
                        answer = NO_CONTEXT_ANSWER
                    else:
                        # Prefer extractive grounding for explicit error/alarm codes
                        extractive = extractive_error_answer(question, retrieved)
                        if extractive:
                            answer = extractive
                        else:
                            messages = build_messages(question, retrieved)
                            answer = generate_answer(messages)
                except Exception as exc:
                    logger.error("Generation error: %s", exc)
                    answer = (
                        "I retrieved documentation, but the language model failed to "
                        f"generate a response ({exc}). Please try again.\n\n"
                        "### Sources\n"
                        + "\n".join(
                            f"- {r.get('source')} — Page {r.get('page')}"
                            for r in retrieved
                        )
                    )
                generation_ms = (time.perf_counter() - t1) * 1000

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Query complete — retrieval=%.0fms generation=%.0fms total=%.0fms sources=%d",
            retrieval_ms,
            generation_ms,
            total_ms,
            len(retrieved),
        )
        return {
            "question": question,
            "answer": answer,
            "sources": retrieved,
            "model": active_model_name(),
            "timings_ms": {
                "retrieval": round(retrieval_ms, 1),
                "generation": round(generation_ms, 1),
                "total": round(total_ms, 1),
            },
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "ready": self.is_ready,
            "documents": self.store.document_count() if self.store.chunks else 0,
            "chunks": self.store.size,
            "model": active_model_name(),
            "manifest": self.store.get_manifest() if self.store.exists() else {},
        }
