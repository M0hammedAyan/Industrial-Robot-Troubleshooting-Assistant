"""Retrieval layer over FAISS + embeddings with model-aware filtering."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.config import RETRIEVAL_THRESHOLD, TOP_K
from src.embeddings import embed_query
from src.grounding import enrich_chunk_metadata, extract_error_codes
from src.logging_utils import get_logger
from src.robot_models import (
    chunk_matches_model,
    detect_query_model,
    is_shared_manual,
)
from src.vector_store import VectorStore

logger = get_logger(__name__)

# Soft boosts applied after cosine similarity (IndexFlatIP on L2-normalized vectors).
MODEL_MATCH_BOOST = 0.12
SHARED_MANUAL_BOOST = 0.03
ERROR_CODE_BOOST = 0.10


class Retriever:
    """Embed queries and return ranked document chunks."""

    def __init__(self, store: VectorStore) -> None:
        self.store = store

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        return_debug: bool = False,
    ):
        """
        Retrieve top-k chunks for a question.

        FAISS index is IndexFlatIP over L2-normalized embeddings, so scores are
        cosine similarities in roughly [-1, 1] (typically ~0–1 for related text).

        Each result dict contains chunk fields plus:
          - score (final ranking score after boosts)
          - raw_score (pure cosine similarity)
          - rank
          - model (inferred when possible)

        When evidence is below RETRIEVAL_THRESHOLD, returns an empty list
        (does NOT fall back to weak unrelated hits).
        """
        k = TOP_K if top_k is None else top_k
        threshold = RETRIEVAL_THRESHOLD if min_score is None else min_score

        if self.store.index is None:
            raise RuntimeError("Vector store is not loaded")

        query_model = detect_query_model(question)
        codes = extract_error_codes(question)
        fetch_k = min(max(k * 8, 20), max(self.store.size, k))
        query_vec = embed_query(question)
        hits = self.store.search(query_vec, top_k=fetch_k)

        scored = []
        for meta, raw_score in hits:
            item = enrich_chunk_metadata(meta)
            text = (item.get("text") or "").upper()
            boost = 0.0

            if query_model:
                if chunk_matches_model(item, query_model):
                    boost += MODEL_MATCH_BOOST
                elif is_shared_manual(item.get("source", "")):
                    boost += SHARED_MANUAL_BOOST
                else:
                    # Strong penalty for other robot models when a model is named
                    boost -= 0.20

            for code in codes:
                if re_code_in_text(code, text):
                    boost = max(boost, ERROR_CODE_BOOST)

            final = float(raw_score) + boost
            scored.append((item, float(raw_score), final))

        scored.sort(key=lambda x: x[2], reverse=True)

        # Prefer model-matching candidates when the query names a model
        if query_model:
            model_scored = [
                t for t in scored if chunk_matches_model(t[0], query_model)
            ]
            shared_scored = [
                t
                for t in scored
                if is_shared_manual(t[0].get("source", ""))
                and t not in model_scored
            ]
            if model_scored:
                scored = model_scored + shared_scored

        candidates_debug = []
        for item, raw, final in scored[: max(k * 3, 10)]:
            candidates_debug.append(
                {
                    "source": item.get("source"),
                    "page": item.get("page"),
                    "model": item.get("model"),
                    "raw_score": round(raw, 4),
                    "score": round(final, 4),
                    "snippet": (item.get("text") or "")[:160],
                }
            )

        logger.info(
            "Retrieved candidates (cosine/IP scores): %s",
            [
                f"{i+1}. score={c['score']:.4f} raw={c['raw_score']:.4f} "
                f"{c['source']} p{c['page']}"
                for i, c in enumerate(candidates_debug[:8])
            ],
        )

        results: List[dict] = []
        seen_keys = set()
        for item, raw, final in scored:
            # Gate on raw cosine similarity so boosts cannot revive weak matches
            if raw < threshold:
                continue
            key = (item.get("source"), item.get("page"), item.get("chunk_id"))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out = dict(item)
            out["raw_score"] = round(raw, 4)
            out["score"] = round(final, 4)
            out["rank"] = len(results) + 1
            results.append(out)
            if len(results) >= k:
                break

        # IMPORTANT: do not fall back to below-threshold hits
        logger.info(
            "Retrieval — model=%s top_k=%d threshold=%.3f returned=%d codes=%s",
            query_model or "none",
            k,
            threshold,
            len(results),
            codes or "none",
        )

        debug: Dict[str, Any] = {
            "question": question,
            "detected_model": query_model,
            "threshold": threshold,
            "score_type": "cosine_similarity (FAISS IndexFlatIP on L2-normalized vectors)",
            "candidates": candidates_debug,
            "final_evidence": [
                {
                    "source": r.get("source"),
                    "page": r.get("page"),
                    "model": r.get("model"),
                    "score": r.get("score"),
                    "raw_score": r.get("raw_score"),
                }
                for r in results
            ],
        }

        if return_debug:
            return results, debug
        return results


def re_code_in_text(code: str, text_upper: str) -> bool:
    import re

    return bool(re.search(rf"\b{re.escape(code.upper())}\b", text_upper))
