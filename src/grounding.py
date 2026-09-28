"""Grounding helpers to reduce unsupported technical claims."""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from src.robot_models import (
    chunk_matches_model,
    detect_query_model,
    infer_model_from_source,
    is_shared_manual,
)

# Universal Robots style + legacy sample codes
ERROR_CODE_RE = re.compile(
    r"\b(?:"
    r"C\d{1,4}(?:[A-Z]\d{0,3})?"  # C173A0, C161, C0
    r"|E[-\s]?\d{2,5}"
    r"|A[-\s]?\d{2,5}"
    r"|ALM[-\s]?\d{2,5}"
    r"|ALARM\s*\d{2,5}"
    r"|ERR(?:OR)?\s*(?:CODE\s*)?\d{2,5}"
    r"|SRVO[-\s]?\d{2,5}"
    r")\b",
    re.IGNORECASE,
)

UNSUPPORTED_ENTITY_RE = re.compile(
    r"\b(?:robot\s+)?xyz\b|\bnasa\b|\bquantum\b",
    re.IGNORECASE,
)

STOPWORDS = {
    "what",
    "when",
    "where",
    "which",
    "how",
    "why",
    "does",
    "do",
    "did",
    "mean",
    "means",
    "the",
    "a",
    "an",
    "is",
    "are",
    "of",
    "for",
    "to",
    "and",
    "or",
    "in",
    "on",
    "with",
    "about",
    "tell",
    "me",
    "please",
    "recommended",
    "recommend",
    "according",
    "available",
    "documentation",
    "document",
    "manual",
    "robot",
    "robots",
    "error",
    "code",
    "codes",
    "alarm",
    "causes",
    "cause",
    "should",
    "can",
    "i",
    "you",
    "this",
    "that",
    "from",
}

# Spec/property terms that must appear in evidence when asked
CRITICAL_SPEC_TERMS = {
    "hydraulic",
    "pressure",
    "payload",
    "temperature",
    "voltage",
    "current",
    "amperage",
    "torque",
    "reach",
    "weight",
    "mass",
    "speed",
    "force",
    "power",
    "frequency",
    "ip",
    "ingress",
}


def extract_error_codes(question: str) -> List[str]:
    """Detect likely error/alarm identifiers in a user question."""
    matches = ERROR_CODE_RE.findall(question or "")
    cleaned: List[str] = []
    for m in matches:
        token = re.sub(r"\s+", "", m.upper())
        token = token.replace("ALARM", "ALM")
        if token not in cleaned:
            cleaned.append(token)
    return cleaned


def context_blob(chunks: Sequence[dict]) -> str:
    return "\n".join((c.get("text") or "") for c in chunks)


def enrich_chunk_metadata(chunk: dict) -> dict:
    """Attach inferred model without mutating the on-disk index schema permanently."""
    item = dict(chunk)
    if not item.get("model"):
        model = infer_model_from_source(item.get("source", ""))
        if model:
            item["model"] = model
    return item


def filter_chunks_for_codes(chunks: Sequence[dict], question: str) -> List[dict]:
    """If the question names error codes, keep chunks that mention them."""
    codes = extract_error_codes(question)
    if not codes:
        return list(chunks)
    matched = []
    for chunk in chunks:
        text = (chunk.get("text") or "").upper()
        for code in codes:
            digits = re.sub(r"\D", "", code)
            if code in text or (len(code) >= 4 and code in text.replace(" ", "")):
                matched.append(chunk)
                break
            # Prefer exact alphanumeric token presence
            if re.search(rf"\b{re.escape(code)}\b", text):
                matched.append(chunk)
                break
            if digits and len(digits) >= 3 and digits in text:
                # weak digit-only match — only if full code also loosely present
                if code[:2] in text or code in text.replace("-", ""):
                    matched.append(chunk)
                    break
    return matched  # empty means code not found — caller should refuse


def content_keywords(question: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+\-]{2,}", question or "")
    out = []
    for w in words:
        low = w.lower()
        if low in STOPWORDS:
            continue
        if detect_query_model(w):
            continue
        if low not in out:
            out.append(low)
    return out


def critical_terms_asked(question: str) -> List[str]:
    q = (question or "").lower()
    return [t for t in CRITICAL_SPEC_TERMS if re.search(rf"\b{re.escape(t)}\b", q)]


def evidence_supports_question(
    question: str, chunks: Sequence[dict]
) -> Tuple[bool, str]:
    """
    Validate that retrieved chunks actually contain evidence for the question.

    Returns (ok, reason_if_not_ok).
    """
    if not chunks:
        return False, "no_chunks"

    blob = context_blob(chunks)
    blob_l = blob.lower()
    blob_u = blob.upper()

    if UNSUPPORTED_ENTITY_RE.search(question or ""):
        # Still allow if docs somehow contain it; otherwise refuse
        if "nasa" in (question or "").lower() and "nasa" not in blob_l:
            return False, "unsupported_entity"
        if "quantum" in (question or "").lower() and "quantum" not in blob_l:
            return False, "unsupported_entity"
        if re.search(r"\bxyz\b", question or "", re.I) and "xyz" not in blob_l:
            return False, "unsupported_entity"

    codes = extract_error_codes(question)
    if codes:
        if not any(
            re.search(rf"\b{re.escape(code)}\b", blob_u) for code in codes
        ):
            return False, "code_not_in_context"

    model = detect_query_model(question)
    if model:
        model_ok = any(chunk_matches_model(c, model) for c in chunks)
        if not model_ok:
            # Shared manuals may still be OK for non-model-specific facts if text mentions model
            shared_ok = any(is_shared_manual(c.get("source", "")) for c in chunks)
            if not shared_ok:
                return False, "model_mismatch"

    # Spec questions require the critical terms to appear in evidence
    critical = critical_terms_asked(question)
    if critical:
        missing = [t for t in critical if t not in blob_l]
        # Allow synonyms for a few cases
        if "payload" in missing and any(
            x in blob_l for x in ("maximum payload", "rated payload", "kg /", "kg)")
        ):
            missing = [t for t in missing if t != "payload"]
        if missing:
            return False, f"missing_terms:{','.join(missing)}"

    # Multi-word technical phrases with synonyms
    PHRASE_SYNONYMS = {
        "protective stop": ["safeguard stop", "safeguard stops", "protective stops"],
        "emergency stop": ["e-stop", "emergency stops"],
        "teach pendant": ["tp-70", "teach pendants"],
        "safety function": ["safety functions"],
        "hydraulic pressure": [],
    }

    phrases = []
    q_l = (question or "").lower()
    for phrase in (
        "hydraulic pressure",
        "protective stop",
        "emergency stop",
        "teach pendant",
        "safety function",
    ):
        if phrase in q_l:
            phrases.append(phrase)
    for phrase in phrases:
        # allow close variants and synonyms
        if phrase in blob_l:
            continue
        synonyms = PHRASE_SYNONYMS.get(phrase, [])
        if any(syn in blob_l for syn in synonyms):
            continue
        parts = phrase.split()
        if not all(p in blob_l for p in parts):
            return False, f"missing_phrase:{phrase}"

    # General keyword overlap — require at least one strong content keyword hit
    # when no error code and no critical terms (e.g. protective stop already handled)
    keywords = content_keywords(question)
    if keywords and not codes and not critical and not phrases:
        hits = sum(1 for k in keywords if k in blob_l)
        if hits == 0:
            return False, "no_keyword_overlap"

    return True, ""


# Phrase synonyms for evidence filtering
_PHRASE_SYNONYMS = {
    "protective stop": ["safeguard stop", "safeguard stops", "protective stops"],
    "emergency stop": ["e-stop", "emergency stops"],
    "teach pendant": ["tp-70", "teach pendants"],
    "safety function": ["safety functions"],
    "hydraulic pressure": [],
}


def filter_evidence_chunks(question: str, chunks: Sequence[dict]) -> List[dict]:
    """Keep only chunks that contribute evidence for this question."""
    model = detect_query_model(question)
    codes = extract_error_codes(question)
    critical = critical_terms_asked(question)
    q_l = (question or "").lower()
    phrases = [
        p
        for p in (
            "hydraulic pressure",
            "protective stop",
            "emergency stop",
            "teach pendant",
            "safety function",
        )
        if p in q_l
    ]

    kept: List[dict] = []
    for chunk in chunks:
        item = enrich_chunk_metadata(chunk)
        text = (item.get("text") or "")
        text_l = text.lower()
        text_u = text.upper()

        if model and not chunk_matches_model(item, model):
            # Allow shared manuals only when they contain model or code evidence
            if not is_shared_manual(item.get("source", "")):
                continue
            if model.lower() not in text_l and not codes:
                continue

        if codes:
            if not any(re.search(rf"\b{re.escape(c)}\b", text_u) for c in codes):
                continue

        if critical:
            if any(t not in text_l for t in critical):
                # keep if phrase parts present for payload-like specs
                if not (
                    "payload" in critical
                    and any(x in text_l for x in ("payload", "maximum payload"))
                ):
                    if not all(t in text_l for t in critical):
                        continue

        if phrases:
            ok_phrase = False
            for phrase in phrases:
                if phrase in text_l or all(p in text_l for p in phrase.split()):
                    ok_phrase = True
                    break
                # Check synonyms
                synonyms = _PHRASE_SYNONYMS.get(phrase, [])
                if any(syn in text_l for syn in synonyms):
                    ok_phrase = True
                    break
            if not ok_phrase:
                continue

        kept.append(item)

    return kept


def extractive_error_answer(question: str, chunks: Sequence[dict]) -> Optional[str]:
    """Build a structured answer from the chunk that defines an error code."""
    codes = extract_error_codes(question)
    if not codes:
        return None

    split_re = re.compile(
        r"(?=(?:\b(?:C\d{1,4}(?:[A-Z]\d{0,3})?|A|E|SRVO|COMM|ALM)[-\s]?\d{0,5}\b))",
        re.IGNORECASE,
    )

    for chunk in chunks:
        text = chunk.get("text") or ""
        upper = text.upper()
        for code in codes:
            if not re.search(rf"\b{re.escape(code)}\b", upper):
                continue

            segments = [s.strip() for s in split_re.split(text) if s.strip()]
            chosen = None
            for seg in segments:
                if re.search(rf"\b{re.escape(code)}\b", seg.upper()):
                    chosen = seg
                    break
            if not chosen:
                chosen = text
            if len(chosen) > 700:
                chosen = chosen[:700].rsplit(" ", 1)[0] + "..."

            sources = [f"- {chunk.get('source')} — Page {chunk.get('page')}"]
            return (
                "### Diagnosis\n"
                f"According to the retrieved documentation, **{code}** is described as:\n\n"
                f"{chosen}\n\n"
                "### Possible Causes\n"
                "See the documented causes in the excerpt above when listed.\n\n"
                "### Recommended Checks\n"
                "Follow the documented checks in the excerpt above when listed.\n\n"
                "### Recommended Procedure\n"
                "Follow the corrective action described in the documentation excerpt.\n\n"
                "### Safety\n"
                "Follow the manufacturer's official safety procedure and site "
                "lockout/tagout requirements before maintenance work.\n\n"
                "### Sources\n"
                + "\n".join(dict.fromkeys(sources))
            )
    return None


def should_refuse_generation(
    question: str, chunks: Sequence[dict], max_score: float, threshold: float
) -> Tuple[bool, str]:
    """Decide whether to refuse instead of calling the LLM."""
    if not chunks:
        return True, "no_chunks"
    if max_score < threshold:
        return True, "low_score"
    ok, reason = evidence_supports_question(question, chunks)
    if not ok:
        return True, reason
    return False, ""
