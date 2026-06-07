from __future__ import annotations

"""Phase 12 + Phase 13 for a legal RAG pipeline.

Input:
- A Phase 11 output record, usually shaped like:
  {
    "query": ...,
    "phase8": ...,
    "retrieval": {...},
    "phase11": {...}
  }

What this file does:
- Phase 12: compute confidence from Phase 10 + Phase 11 signals
- Phase 13: run final safety / validation checks
- Return the final user-facing result

This version stays small and practical.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


JsonLike = Union[Dict[str, Any], List[Any]]

SECTION_REF_PATTERN = re.compile(
    r"\b(?:section|sec\.?|s\.|subsection|sub-section|clause)\s*"
    r"(\d+[a-z]?)(?:\s*\(\s*([0-9a-z]+)\s*\))?",
    flags=re.IGNORECASE,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _norm_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _norm_key(value: Any) -> str:
    return _norm_text(value).lower()


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        text = _norm_text(item)
        if not text:
            continue
        key = _norm_key(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _merge_notes(items: Sequence[str]) -> List[str]:
    """Deduplicate exact and near-duplicate messages without being opinionated."""
    out: List[str] = []
    seen_keys: List[str] = []
    for raw in items:
        text = _norm_text(raw)
        if not text:
            continue
        key = _norm_key(text)
        if key in seen_keys:
            continue
        # Remove obvious near-duplicates where one note is contained in another.
        duplicate = False
        for existing in out:
            ex = _norm_key(existing)
            if key in ex or ex in key:
                duplicate = True
                break
        if duplicate:
            continue
        seen_keys.append(key)
        out.append(text)
    return out


def _load_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _as_dict(obj: Any) -> Dict[str, Any]:
    return obj if isinstance(obj, dict) else {}


def _as_list(obj: Any) -> List[Any]:
    return obj if isinstance(obj, list) else []


def _phase11_block(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return the Phase 11 payload, whether the input is nested or already flat."""
    if "phase11" in record and isinstance(record["phase11"], dict):
        return record["phase11"]
    return record


def _retrieval_block(record: Dict[str, Any]) -> Dict[str, Any]:
    return _as_dict(record.get("retrieval"))


def _extract_section_tokens(text: Any) -> List[str]:
    raw = _norm_text(text)
    if not raw:
        return []

    tokens: List[str] = []
    for match in SECTION_REF_PATTERN.finditer(raw):
        base = _norm_text(match.group(1)).upper()
        sub = _norm_text(match.group(2))
        tokens.append(f"Section {base}{f'({sub})' if sub else ''}")
    return _dedupe_keep_order(tokens)


def _selected_section_tokens(phase11: Dict[str, Any]) -> List[str]:
    sr = _as_dict(phase11.get("structured_reasoning"))
    selected_section = _norm_text(sr.get("selected_section"))
    if not selected_section:
        return []
    return _extract_section_tokens(selected_section)


def _top_rerank_score(retrieval: Dict[str, Any]) -> float:
    """Phase 10 reranker signal: use the top score from the reranked list."""
    ranked = _as_list(retrieval.get("results_with_global_rerank"))
    scores: List[float] = []
    for row in ranked:
        if isinstance(row, dict):
            scores.append(_to_float(row.get("score"), 0.0))
    if scores:
        return _clip(max(scores))

    # Fallback only if reranked results are absent.
    plain = _as_list(retrieval.get("results_without_global_rerank"))
    scores = []
    for row in plain:
        if isinstance(row, dict):
            scores.append(_to_float(row.get("score"), 0.0))
    return _clip(max(scores) if scores else 0.0)


def _retrieval_strength(retrieval: Dict[str, Any]) -> float:
    ranked = _as_list(retrieval.get("results_with_global_rerank"))
    if not ranked:
        ranked = _as_list(retrieval.get("results_without_global_rerank"))

    scores: List[float] = []
    for row in ranked:
        if isinstance(row, dict):
            scores.append(_to_float(row.get("score"), 0.0))
    return _clip(max(scores) if scores else 0.0)


def _ordered_citations(phase11: Dict[str, Any]) -> List[str]:
    citations = _dedupe_keep_order([_norm_text(c) for c in _as_list(phase11.get("citations"))])
    if not citations:
        return []

    selected_tokens = _selected_section_tokens(phase11)
    if not selected_tokens:
        return citations

    # Prefer citations that mention the selected section, but do not discard
    # the rest unless we have a better match.
    ranked: List[Tuple[int, str]] = []
    for cit in citations:
        key = _norm_key(cit)
        score = 0
        for token in selected_tokens:
            token_key = _norm_key(token)
            if token_key == key:
                score += 3
            elif token_key in key or key in token_key:
                score += 2
            else:
                # Match on bare section number if available.
                section_tokens = _extract_section_tokens(cit)
                if any(_norm_key(x) == token_key for x in section_tokens):
                    score += 2
        ranked.append((score, cit))

    ranked.sort(key=lambda item: (item[0], -citations.index(item[1])), reverse=True)
    ordered = [cit for _, cit in ranked]
    return _dedupe_keep_order(ordered)


_UNCERTAINTY_TO_REASONING = {
    "low": 0.92,
    "medium": 0.65,
    "high": 0.30,
}

_UNCERTAINTY_TO_CERTAINTY = {
    "low": 1.00,
    "medium": 0.70,
    "high": 0.35,
}


_RISK_REASONS = {
    "section_not_in_corpus": "Requested section is not present in the indexed corpus, so no substantive answer is shown.",
    "unsafe": "The retrieved evidence is not sufficient to safely support this answer.",
    "unverified": "The answer could not be verified against the retrieved evidence.",
    "missing_citations": "The answer does not have enough citation support from retrieved evidence.",
    "fallback": "The answer is grounded in retrieved evidence, but deterministic fallback was used after generation did not pass grounding checks.",
    "repaired": "The answer passed grounding checks after one repair pass, so some caution remains.",
    "partial": "The answer is supported only partially by retrieved evidence.",
    "high_uncertainty": "The applicable legal branch is uncertain because competing provisions or factual conditions remain unresolved.",
    "confidence_medium": "The answer is grounded, but confidence is not high enough to treat it as low risk.",
    "confidence_high": "Confidence is too low to provide this as a reliable final answer.",
    "low": "Evidence, citations, and reasoning are aligned well enough for the final answer view.",
}


def _average(values: Sequence[float], default: float = 0.0) -> float:
    cleaned = [float(value) for value in values if isinstance(value, (int, float))]
    if not cleaned:
        return default
    return _clip(sum(cleaned) / len(cleaned))


def _retrieval_rows(retrieval: Dict[str, Any]) -> List[Dict[str, Any]]:
    ranked = _as_list(retrieval.get("results_with_global_rerank"))
    if not ranked:
        ranked = _as_list(retrieval.get("results_without_global_rerank"))
    return [row for row in ranked if isinstance(row, dict)]


def _selected_rows(phase11: Dict[str, Any], retrieval: Dict[str, Any]) -> List[Dict[str, Any]]:
    selected_ids = [
        _norm_text(chunk_id)
        for chunk_id in _as_list(phase11.get("selected_chunk_ids"))
        if _norm_text(chunk_id)
    ]
    if not selected_ids:
        return []

    rows_by_chunk_id: Dict[str, Dict[str, Any]] = {}
    for row in _retrieval_rows(retrieval):
        chunk_id = _norm_text(row.get("chunk_id"))
        if chunk_id and chunk_id not in rows_by_chunk_id:
            rows_by_chunk_id[chunk_id] = row

    return [rows_by_chunk_id[chunk_id] for chunk_id in selected_ids if chunk_id in rows_by_chunk_id]


def _row_semrank_score(row: Dict[str, Any]) -> Optional[float]:
    candidates: List[float] = []

    semrank = _as_dict(row.get("semrank"))
    for key in ("pre_rerank_score", "score", "best_similarity", "retrieval_score"):
        if key in semrank:
            candidates.append(_to_float(semrank.get(key), 0.0))

    source_scores = _as_dict(row.get("source_scores"))
    for key, value in source_scores.items():
        if "semrank" in str(key).lower():
            candidates.append(_to_float(value, 0.0))

    if not candidates:
        return None
    return _clip(max(candidates))


def _row_concept_coverage(row: Dict[str, Any]) -> float:
    return _clip(_to_float(_as_dict(row.get("concept_coverage")).get("coverage_ratio"), 0.0))


def _section_key(value: Any) -> str:
    tokens = _extract_section_tokens(value)
    if tokens:
        return _norm_key(tokens[0])
    return _norm_key(value)


def _section_label_from_row(row: Dict[str, Any]) -> str:
    section_number = _norm_text(row.get("section_number"))
    if section_number:
        return f"Section {section_number}"
    return _norm_text(row.get("citation"))


def _evidence_strength(phase11: Dict[str, Any], retrieval: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    rows = _selected_rows(phase11, retrieval)

    chunk_scores = [_clip(_to_float(row.get("score"), 0.0)) for row in rows]
    semrank_scores = [_row_semrank_score(row) for row in rows]
    coverage_scores = [_row_concept_coverage(row) for row in rows]

    avg_chunk = _average(chunk_scores)
    avg_semrank = _average(semrank_scores, default=avg_chunk)
    avg_coverage = _average(coverage_scores)

    score = _clip(
        0.55 * avg_chunk
        + 0.35 * avg_semrank
        + 0.10 * avg_coverage
    )

    selected_ids = [
        _norm_text(chunk_id)
        for chunk_id in _as_list(phase11.get("selected_chunk_ids"))
        if _norm_text(chunk_id)
    ]
    return score, {
        "avg_selected_chunk_score": round(avg_chunk, 4),
        "avg_selected_semrank_score": round(avg_semrank, 4),
        "avg_selected_concept_coverage": round(avg_coverage, 4),
        "selected_chunk_count": len(selected_ids),
        "matched_selected_chunk_count": len(rows),
        "selected_semrank_count": len([score for score in semrank_scores if isinstance(score, (int, float))]),
    }


def _structured_sections(phase11: Dict[str, Any]) -> List[Dict[str, Any]]:
    sr = _as_dict(phase11.get("structured_reasoning"))
    return [row for row in _as_list(sr.get("sections_analyzed")) if isinstance(row, dict)]


def _selected_condition_match_score(phase11: Dict[str, Any], retrieval: Dict[str, Any]) -> float:
    sr = _as_dict(phase11.get("structured_reasoning"))
    sections = _structured_sections(phase11)
    selected_key = _section_key(sr.get("selected_section"))

    if selected_key:
        matches = [
            _to_float(row.get("condition_match_score"), 0.0)
            for row in sections
            if _section_key(row.get("section")) == selected_key
        ]
        if matches:
            return _clip(max(matches))

    selected_section_keys = {
        _section_key(_section_label_from_row(row))
        for row in _selected_rows(phase11, retrieval)
        if _section_label_from_row(row)
    }
    fallback_matches = [
        _to_float(row.get("condition_match_score"), 0.0)
        for row in sections
        if _section_key(row.get("section")) in selected_section_keys
    ]
    if fallback_matches:
        return _average(fallback_matches, default=0.65)

    return 0.65


def _ambiguity_gap_score(phase11: Dict[str, Any], selected_score: float) -> Tuple[float, Dict[str, Any]]:
    sr = _as_dict(phase11.get("structured_reasoning"))
    validation = _as_dict(phase11.get("validation"))
    selected_key = _section_key(sr.get("selected_section"))

    competitors: List[float] = []
    for row in _structured_sections(phase11):
        if selected_key and _section_key(row.get("section")) == selected_key:
            continue
        competitors.append(_clip(_to_float(row.get("condition_match_score"), 0.0)))

    if not competitors:
        return 1.0, {"gap": None, "highest_competing_score": None}

    highest_competing = max(competitors)
    gap = selected_score - highest_competing
    if gap >= 0.15:
        score = 1.0
    elif gap >= 0.08:
        score = 0.75
    elif gap >= 0.03:
        score = 0.50
    else:
        score = 0.30

    mitigated = False
    if (
        score < 0.60
        and bool(validation.get("answer_verified", False))
        and _norm_key(validation.get("completeness")) == "complete"
        and _norm_key(sr.get("uncertainty")) == "low"
        and not bool(validation.get("fallback_used", False))
        and not bool(validation.get("repaired", False))
    ):
        score = 0.60
        mitigated = True

    return score, {
        "gap": round(gap, 4),
        "highest_competing_score": round(highest_competing, 4),
        "ambiguity_mitigated_by_grounding": mitigated,
    }


def _uncertainty_score(phase11: Dict[str, Any]) -> float:
    uncertainty = _norm_key(_as_dict(phase11.get("structured_reasoning")).get("uncertainty"))
    return _UNCERTAINTY_TO_CERTAINTY.get(uncertainty, 0.65)


def _reasoning_certainty(phase11: Dict[str, Any], retrieval: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    selected_score = _selected_condition_match_score(phase11, retrieval)
    gap_score, gap_details = _ambiguity_gap_score(phase11, selected_score)
    uncertainty = _uncertainty_score(phase11)

    score = _clip(
        0.70 * selected_score
        + 0.20 * gap_score
        + 0.10 * uncertainty
    )

    return score, {
        "selected_condition_match_score": round(selected_score, 4),
        "ambiguity_gap_score": round(gap_score, 4),
        "uncertainty_score": round(uncertainty, 4),
        **gap_details,
    }


def _grounding_certainty(phase11: Dict[str, Any]) -> float:
    validation = _as_dict(phase11.get("validation"))
    guardrail_strength = validation.get("guardrail_grounding_strength")
    if isinstance(guardrail_strength, (int, float)):
        return _clip(float(guardrail_strength))

    answer_verified = bool(validation.get("answer_verified", False))
    completeness = _norm_key(validation.get("completeness"))
    if answer_verified and completeness == "complete":
        return 1.0
    if answer_verified:
        return 0.72
    return 0.25


def _confidence_caps(phase11: Dict[str, Any], citations: Sequence[str]) -> List[Dict[str, Any]]:
    validation = _as_dict(phase11.get("validation"))
    completeness = _norm_key(validation.get("completeness"))
    support_status = _norm_key(phase11.get("support_status"))

    caps: List[Dict[str, Any]] = []
    if support_status == "section_not_in_corpus":
        caps.append({"reason": "section_not_in_corpus", "max_score": 0.25})
    if not bool(validation.get("answer_verified", False)):
        caps.append({"reason": "unverified", "max_score": 0.35})
    if completeness == "unsafe":
        caps.append({"reason": "unsafe", "max_score": 0.25})
    if completeness == "partial":
        caps.append({"reason": "partial", "max_score": 0.70})
    if bool(validation.get("fallback_used", False)):
        caps.append({"reason": "fallback", "max_score": 0.65})
    if bool(validation.get("repaired", False)):
        caps.append({"reason": "repaired", "max_score": 0.80})
    if not citations:
        caps.append({"reason": "missing_citations", "max_score": 0.40})
    return caps


def _grounded_confidence_floor(phase11: Dict[str, Any], citations: Sequence[str], applied_caps: Sequence[Dict[str, Any]]) -> Optional[float]:
    validation = _as_dict(phase11.get("validation"))
    sr = _as_dict(phase11.get("structured_reasoning"))

    if applied_caps:
        return None
    if not bool(validation.get("answer_verified", False)):
        return None
    if _norm_key(validation.get("completeness")) != "complete":
        return None
    if bool(validation.get("fallback_used", False)) or bool(validation.get("repaired", False)):
        return None
    if not citations:
        return None
    if not _norm_text(sr.get("selected_section")):
        return None
    if not _as_list(phase11.get("selected_chunk_ids")):
        return None
    return 0.72


def _reasoning_strength(phase11: Dict[str, Any]) -> float:
    sr = _as_dict(phase11.get("structured_reasoning"))
    validation = _as_dict(phase11.get("validation"))

    guardrail_strength = validation.get("guardrail_grounding_strength")
    if isinstance(guardrail_strength, (int, float)):
        score = _clip(float(guardrail_strength))
        if bool(validation.get("fallback_used", False)):
            score -= 0.05
        if bool(validation.get("repaired", False)):
            score -= 0.03
        return _clip(score)

    uncertainty = _norm_key(sr.get("uncertainty"))
    score = _UNCERTAINTY_TO_REASONING.get(uncertainty, 0.65)

    if bool(sr.get("overrides_score_selection", False)):
        score -= 0.05
    if bool(validation.get("answer_verified", False)):
        score += 0.05
    if _norm_key(validation.get("completeness")) == "complete":
        score += 0.05

    return _clip(score)


def _confidence_label(score: float) -> str:
    if score >= 0.80:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


def _section_consensus(rows: Sequence[Dict[str, Any]], limit: int = 5) -> float:
    top_rows = [row for row in rows[:limit] if isinstance(row, dict)]
    if not top_rows:
        return 0.0

    sections = [
        _section_key(_section_label_from_row(row))
        for row in top_rows
        if _section_label_from_row(row)
    ]
    if not sections:
        return 0.0

    counts: Dict[str, int] = {}
    for section in sections:
        counts[section] = counts.get(section, 0) + 1
    return _clip(max(counts.values()) / len(sections))


def _selector_retention(phase11: Dict[str, Any], retrieval: Dict[str, Any]) -> float:
    selected_ids = [
        _norm_text(chunk_id)
        for chunk_id in _as_list(phase11.get("selected_chunk_ids"))
        if _norm_text(chunk_id)
    ]
    if not selected_ids:
        return 0.0

    retrieved_ids = {
        _norm_text(row.get("chunk_id"))
        for row in _retrieval_rows(retrieval)
        if _norm_text(row.get("chunk_id"))
    }
    if not retrieved_ids:
        return 0.0

    retained = len([chunk_id for chunk_id in selected_ids if chunk_id in retrieved_ids])
    return _clip(retained / len(selected_ids))


def _retrieval_confidence(
    phase11: Dict[str, Any],
    retrieval: Dict[str, Any],
    evidence_components: Dict[str, Any],
) -> Dict[str, Any]:
    rows = _retrieval_rows(retrieval)
    top_rerank_score = _retrieval_strength(retrieval)
    selected_evidence_score = _clip(_to_float(evidence_components.get("avg_selected_chunk_score"), 0.0))
    semantic_match = _clip(_to_float(evidence_components.get("avg_selected_semrank_score"), selected_evidence_score))
    concept_coverage = _clip(_to_float(evidence_components.get("avg_selected_concept_coverage"), 0.0))
    section_consensus = _section_consensus(rows)
    selector_retention = _selector_retention(phase11, retrieval)

    score = _clip(
        0.40 * top_rerank_score
        + 0.30 * selected_evidence_score
        + 0.20 * semantic_match
        + 0.10 * concept_coverage
        + 0.05 * section_consensus
        + 0.05 * selector_retention
    )

    return {
        "score": round(score, 4),
        "label": _confidence_label(score),
        "components": {
            "top_rerank_score": round(top_rerank_score, 4),
            "selected_evidence_score": round(selected_evidence_score, 4),
            "semantic_match": round(semantic_match, 4),
            "concept_coverage": round(concept_coverage, 4),
            "section_consensus": round(section_consensus, 4),
            "selector_retention": round(selector_retention, 4),
        },
        "formula": (
            "0.40*top_rerank_score + 0.30*selected_evidence_score + "
            "0.20*semantic_match + 0.10*concept_coverage + "
            "0.05*section_consensus + 0.05*selector_retention"
        ),
        "explanation": (
            "Retrieval confidence estimates whether the retrieved and selected chunks "
            "contain relevant legal support. It is separate from final answer confidence."
        ),
    }


# -----------------------------------------------------------------------------
# Phase 12
# -----------------------------------------------------------------------------

def compute_phase12(record: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the final confidence score using Phase 10 + Phase 11 outputs."""
    phase11 = _phase11_block(record)
    retrieval = _retrieval_block(record)
    citations = _ordered_citations(phase11)

    evidence_score, evidence_components = _evidence_strength(phase11, retrieval)
    reasoning_score, reasoning_components = _reasoning_certainty(phase11, retrieval)
    grounding_score = _grounding_certainty(phase11)
    retrieval_confidence = _retrieval_confidence(phase11, retrieval, evidence_components)

    raw_confidence = _clip(evidence_score * reasoning_score * grounding_score)
    applied_caps = _confidence_caps(phase11, citations)
    confidence_floor = _grounded_confidence_floor(phase11, citations, applied_caps)
    floor_adjusted_confidence = (
        max(raw_confidence, confidence_floor)
        if isinstance(confidence_floor, (int, float))
        else raw_confidence
    )
    capped_confidence = min(
        [floor_adjusted_confidence] + [_clip(_to_float(cap.get("max_score"), 1.0)) for cap in applied_caps]
    )
    confidence = round(_clip(capped_confidence), 4)

    for cap in applied_caps:
        if "max_score" in cap:
            cap["max_score"] = round(_clip(_to_float(cap.get("max_score"), 1.0)), 4)

    return {
        # Compatibility keys retained for existing consumers.
        "retrieval": round(evidence_score, 4),
        "reranker": round(evidence_components.get("avg_selected_chunk_score", 0.0), 4),
        "citation": round(1.0 if citations else 0.0, 4),
        "reasoning": round(reasoning_score, 4),
        "evidence_strength": round(evidence_score, 4),
        "reasoning_certainty": round(reasoning_score, 4),
        "grounding_certainty": round(grounding_score, 4),
        "raw_score": round(raw_confidence, 4),
        "confidence_floor": round(confidence_floor, 4) if isinstance(confidence_floor, (int, float)) else None,
        "floor_adjusted_score": round(floor_adjusted_confidence, 4),
        "applied_caps": applied_caps,
        "components": {
            "evidence": evidence_components,
            "reasoning": reasoning_components,
            "grounding": {
                "guardrail_grounding_strength": round(grounding_score, 4),
            },
        },
        "retrieval_confidence": retrieval_confidence,
        "score": confidence,
        "label": _confidence_label(confidence),
        "formula": (
            "min((0.55*avg_selected_chunk_score + 0.35*avg_selected_semrank_score "
            "+ 0.10*avg_selected_concept_coverage) * "
            "(0.70*selected_condition_match_score + 0.20*ambiguity_gap_score "
            "+ 0.10*uncertainty_score) * grounding_certainty, applicable_caps)"
        ),
        "terms": {
            "evidence_strength": "Average selected chunk retrieval score, SemRank score, and concept coverage.",
            "reasoning_certainty": "Selected condition match, ambiguity gap, and structured reasoning uncertainty.",
            "grounding_certainty": "Phase 11 retrieved-evidence guardrail grounding strength.",
            "confidence_floor": "Minimum score for verified, complete, grounded answers when no safety caps apply.",
            "applied_caps": "Validation caps that prevent confidence from exceeding known safety limits.",
        },
        "explanation": (
            "Confidence is calculated from selected evidence, reasoning certainty, "
            "and grounding certainty, then capped by validation safety conditions."
        ),
    }


def _citation_strength(phase11: Dict[str, Any]) -> float:
    citations = _ordered_citations(phase11)
    validation = _as_dict(phase11.get("validation"))
    completeness = _norm_key(validation.get("completeness"))
    answer_verified = bool(validation.get("answer_verified", False))

    if not citations:
        return 0.0
    if completeness == "unsafe":
        return 0.35
    if not answer_verified:
        return 0.70
    return 1.0


def _base_risk_from_confidence(confidence: float) -> Tuple[str, str]:
    if confidence >= 0.80:
        return "low", _RISK_REASONS["low"]
    if confidence >= 0.60:
        return "medium", _RISK_REASONS["confidence_medium"]
    return "high", _RISK_REASONS["confidence_high"]


def _raise_risk_at_least(current: str, minimum: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    if order.get(current, 0) >= order.get(minimum, 0):
        return current
    return minimum


def _derive_risk(
    confidence: float,
    support_status: str,
    completeness: str,
    answer_verified: bool,
    citations: Sequence[str],
    fallback_used: bool,
    repaired: bool,
    uncertainty: str,
) -> Tuple[str, str]:
    if support_status == "section_not_in_corpus":
        return "high", _RISK_REASONS["section_not_in_corpus"]
    if completeness == "unsafe":
        return "high", _RISK_REASONS["unsafe"]
    if not answer_verified:
        return "high", _RISK_REASONS["unverified"]
    if not citations:
        return "high", _RISK_REASONS["missing_citations"]

    risk_level, risk_reason = _base_risk_from_confidence(confidence)

    medium_overrides = [
        (fallback_used, "fallback"),
        (repaired, "repaired"),
        (completeness == "partial", "partial"),
        (uncertainty == "high", "high_uncertainty"),
    ]
    for condition, reason_key in medium_overrides:
        if condition:
            return _raise_risk_at_least(risk_level, "medium"), _RISK_REASONS[reason_key]

    return risk_level, risk_reason


# -----------------------------------------------------------------------------
# Phase 13
# -----------------------------------------------------------------------------

def validate_phase13(record: Dict[str, Any], phase12: Dict[str, Any]) -> Dict[str, Any]:
    """Run final strict checks before showing the answer to the user."""
    phase11 = _phase11_block(record)

    citations = _ordered_citations(phase11)
    validation = _as_dict(phase11.get("validation"))
    sr = _as_dict(phase11.get("structured_reasoning"))
    support_status = _norm_key(phase11.get("support_status"))

    summary_answer = _norm_text(phase11.get("summary_answer"))
    detailed_answer = _norm_text(phase11.get("detailed_answer"))
    final_answer = _norm_text(phase11.get("final_answer"))

    completeness = _norm_key(validation.get("completeness")) or "partial"
    answer_verified = bool(validation.get("answer_verified", False))
    fallback_used = bool(validation.get("fallback_used", False))
    repaired = bool(validation.get("repaired", False))
    confidence = _to_float(phase12.get("score"), 0.0)
    uncertainty = _norm_key(sr.get("uncertainty")) or "medium"
    overrides = bool(sr.get("overrides_score_selection", False))

    issues: List[str] = []
    warnings: List[str] = []

    if not citations:
        issues.append("missing citations")
    if not answer_verified:
        issues.append("phase11 did not verify the answer")
    if completeness == "unsafe":
        issues.append("phase11 marked the answer unsafe")

    # Special case: when Phase 11 says the requested section is not present,
    # do not surface any generated answer text. Show only the warning.
    if support_status == "section_not_in_corpus":
        missing_sections = _dedupe_keep_order([
            _norm_text(x)
            for x in _as_list(phase11.get("missing_from_corpus"))
            if _norm_text(x)
        ])
        if not missing_sections:
            missing_sections = _dedupe_keep_order([
                _norm_text(x)
                for x in _as_list(_as_dict(phase11.get("validation")).get("missing_from_corpus"))
                if _norm_text(x)
            ])
        warning = (
            f"{', '.join(missing_sections) if missing_sections else 'The requested section'} is not present in the indexed corpus."
        )
        warnings.append(warning)
        warnings = _merge_notes(warnings)
        return {
            "answer": warning,
            "summary_answer": "",
            "detailed_answer": "",
            "final_answer": "",
            "display_answer_type": "warning",
            "citations": [],
            "confidence": phase12,
            "risk_level": "high",
            "risk_reason": _RISK_REASONS["section_not_in_corpus"],
            "validation": {
                "answer_verified": answer_verified,
                "completeness": "unsafe",
                "warnings": warnings,
                "verification_issues": _merge_notes(issues + ["requested section not present in corpus"]),
                "should_show": False,
            },
        }

    # Bring through Phase 11 warnings, but deduplicate aggressively.
    raw_phase11_warnings = [
        _norm_text(x)
        for x in _as_list(phase11.get("warnings"))
        if _norm_text(x)
    ]
    warnings.extend(raw_phase11_warnings)

    if uncertainty == "high":
        warnings.append("reasoning uncertainty is high")
    elif uncertainty == "medium" and overrides:
        warnings.append(
            "Condition-based reasoning selected a different section than the highest-scored retrieval hit."
        )

    warnings = _merge_notes(warnings)
    issues = _merge_notes(issues)

    risk_level, risk_reason = _derive_risk(
        confidence=confidence,
        support_status=support_status,
        completeness=completeness,
        answer_verified=answer_verified,
        citations=citations,
        fallback_used=fallback_used,
        repaired=repaired,
        uncertainty=uncertainty,
    )

    if risk_level == "high":
        display_answer_type = "summary"
        answer = (
            summary_answer
            or detailed_answer
            or final_answer
            or "The evidence is not strong enough to provide a reliable answer."
        )
    elif risk_level == "medium":
        display_answer_type = "detailed"
        answer = detailed_answer or final_answer or summary_answer
    else:
        display_answer_type = "detailed"
        answer = final_answer or detailed_answer or summary_answer

    answer = _norm_text(answer)

    matched_sections = _dedupe_keep_order([
        _norm_text(x)
        for x in _as_list(phase11.get("matched_sections"))
        if _norm_text(x)
    ])
    selected_section = _norm_text(_as_dict(phase11.get("structured_reasoning")).get("selected_section"))

    if selected_section and matched_sections and selected_section not in matched_sections:
        warnings.append("selected section is not in the matched sections list")
        warnings = _merge_notes(warnings)

    return {
        "answer": answer,
        "summary_answer": summary_answer,
        "detailed_answer": detailed_answer,
        "final_answer": final_answer,
        "display_answer_type": display_answer_type,
        "citations": citations,
        "confidence": phase12,
        "risk_level": risk_level,
        "risk_reason": risk_reason,
        "validation": {
            "answer_verified": answer_verified,
            "completeness": completeness,
            "warnings": warnings,
            "verification_issues": issues,
            "should_show": risk_level != "high",
        },
    }


# -----------------------------------------------------------------------------
# Combined runner
# -----------------------------------------------------------------------------

def run_pipeline(record: Dict[str, Any]) -> Dict[str, Any]:
    phase11 = _phase11_block(record)
    phase12 = compute_phase12(record)
    phase13 = validate_phase13(record, phase12)
    selected_chunk_ids = _as_list(phase11.get("selected_chunk_ids"))
    answer_confidence = dict(phase13["confidence"])
    retrieval_confidence = answer_confidence.pop("retrieval_confidence", phase12.get("retrieval_confidence"))
    return {
        "answer": phase13["answer"],
        "summary_answer": phase13["summary_answer"],
        "detailed_answer": phase13["detailed_answer"],
        "final_answer": phase13["final_answer"],
        "display_answer_type": phase13["display_answer_type"],
        "citations": phase13["citations"],
        "confidence": answer_confidence,
        "retrieval_confidence": retrieval_confidence,
        "risk_level": phase13["risk_level"],
        "risk_reason": phase13["risk_reason"],
        "validation": phase13["validation"],
        "selected_chunk_ids": selected_chunk_ids,
    }


def process_json(payload: Any) -> Any:
    if isinstance(payload, list):
        return [run_pipeline(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return run_pipeline(payload)
    raise TypeError("Input must be a dict or a list of dicts containing Phase 11 output JSON.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 12 and Phase 13 on Phase 11 output JSON.")
    parser.add_argument("--input_json", help="Path to the JSON file")
    parser.add_argument("-o", "--output", help="Optional output JSON file path")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print to stdout")
    args = parser.parse_args()

    payload = _load_json(args.input_json)
    result = process_json(payload)

    if args.output:
        _save_json(args.output, result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



'''
python validation/validate_1.py \
  --input_json reasoning/res_7__4.json \
  --output validation/final_7__1.json \
  --pretty
  

'''
