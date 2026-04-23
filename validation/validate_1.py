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
from typing import Any, Dict, List, Sequence, Tuple, Union


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


def _reasoning_strength(phase11: Dict[str, Any]) -> float:
    sr = _as_dict(phase11.get("structured_reasoning"))
    validation = _as_dict(phase11.get("validation"))

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


# -----------------------------------------------------------------------------
# Phase 12
# -----------------------------------------------------------------------------

def compute_phase12(record: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the final confidence score using Phase 10 + Phase 11 outputs."""
    phase11 = _phase11_block(record)
    retrieval = _retrieval_block(record)

    retrieval_score = _retrieval_strength(retrieval)
    reranker_score = _top_rerank_score(retrieval)  # Phase 10 score
    citation_score = _citation_strength(phase11)
    reasoning_score = _reasoning_strength(phase11)

    confidence = round(
        0.4 * retrieval_score
        + 0.3 * reranker_score
        + 0.2 * citation_score
        + 0.1 * reasoning_score,
        4,
    )

    return {
        "retrieval": round(retrieval_score, 4),
        "reranker": round(reranker_score, 4),
        "citation": round(citation_score, 4),
        "reasoning": round(reasoning_score, 4),
        "score": confidence,
        "label": _confidence_label(confidence),
        "formula": "0.4*retrieval + 0.3*reranker + 0.2*citation + 0.1*reasoning",
        "terms": {
            "retrieval": "Top score from Phase 9/10 retrieval list",
            "reranker": "Top score from results_with_global_rerank (Phase 10)",
            "citation": "Grounding strength from Phase 11 citations + validation",
            "reasoning": "Phase 11 structured reasoning uncertainty mapped to a score",
        },
        "explanation": (
            "Confidence is a weighted blend of retrieval strength, reranker score, "
            "citation grounding, and structured reasoning certainty."
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
    if confidence < 0.45:
        issues.append("confidence too low")

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
            "risk_reason": "Requested section is missing from the corpus, so no substantive answer is shown.",
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

    # Keep risk and confidence separate on purpose:
    # - confidence = evidence certainty
    # - risk = whether the answer is safe enough to display as the main answer
    if issues:
        risk_level = "high"
        display_answer_type = "summary"
        answer = (
            summary_answer
            or detailed_answer
            or final_answer
            or "The evidence is not strong enough to provide a reliable answer."
        )
        risk_reason = "Validation failed because one or more hard checks did not pass."
    elif confidence < 0.70 or completeness == "partial" or uncertainty == "high":
        risk_level = "medium"
        display_answer_type = "detailed"
        answer = detailed_answer or final_answer or summary_answer
        if overrides or uncertainty == "medium":
            risk_reason = "Evidence is strong, but the applicable branch still depends on factual conditions."
        else:
            risk_reason = "Confidence is not high enough for the final answer view."
    else:
        risk_level = "low"
        display_answer_type = "detailed"
        answer = final_answer or detailed_answer or summary_answer
        risk_reason = "Evidence, citations, and reasoning are aligned well enough for the final answer view."

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
    return {
        "answer": phase13["answer"],
        "summary_answer": phase13["summary_answer"],
        "detailed_answer": phase13["detailed_answer"],
        "final_answer": phase13["final_answer"],
        "display_answer_type": phase13["display_answer_type"],
        "citations": phase13["citations"],
        "confidence": phase13["confidence"],
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