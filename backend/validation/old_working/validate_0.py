
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
- Phase 12: compute confidence from Phase 11 / retrieval output
- Phase 13: run final safety checks
- Return the final user-facing result

This version stays simple on purpose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union


JsonLike = Union[Dict[str, Any], List[Any]]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _norm_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


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
        if not text or text in seen:
            continue
        seen.add(text)
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


def _top_rerank_score(retrieval: Dict[str, Any]) -> float:
    """Phase 10 reranker signal: use the top score from the reranked list."""
    ranked = _as_list(retrieval.get("results_with_global_rerank"))
    scores: List[float] = []
    for row in ranked:
        if isinstance(row, dict):
            scores.append(_to_float(row.get("score"), 0.0))
    if scores:
        return _clip(max(scores))

    # Fallbacks only if reranked results are absent.
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


def _citation_strength(phase11: Dict[str, Any]) -> float:
    citations = _dedupe_keep_order([_norm_text(c) for c in _as_list(phase11.get("citations"))])
    validation = _as_dict(phase11.get("validation"))
    completeness = _norm_text(validation.get("completeness")).lower()
    answer_verified = bool(validation.get("answer_verified", False))

    if not citations:
        return 0.0
    if completeness == "unsafe":
        return 0.35
    if not answer_verified:
        return 0.70
    return 1.0


_UNCERTAINTY_TO_REASONING = {
    "low": 0.92,
    "medium": 0.65,
    "high": 0.30,
}


def _reasoning_strength(phase11: Dict[str, Any]) -> float:
    sr = _as_dict(phase11.get("structured_reasoning"))
    validation = _as_dict(phase11.get("validation"))

    uncertainty = _norm_text(sr.get("uncertainty")).lower()
    score = _UNCERTAINTY_TO_REASONING.get(uncertainty, 0.65)

    if bool(sr.get("overrides_score_selection", False)):
        score -= 0.05
    if bool(validation.get("answer_verified", False)):
        score += 0.05
    if _norm_text(validation.get("completeness")).lower() == "complete":
        score += 0.05

    return _clip(score)


# -----------------------------------------------------------------------------
# Phase 12
# -----------------------------------------------------------------------------

def compute_phase12(record: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the final confidence score using Phase 11 + Phase 10 outputs."""
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

    if confidence >= 0.80:
        label = "high"
    elif confidence >= 0.60:
        label = "medium"
    else:
        label = "low"

    return {
        "retrieval": round(retrieval_score, 4),
        "reranker": round(reranker_score, 4),
        "citation": round(citation_score, 4),
        "reasoning": round(reasoning_score, 4),
        "score": confidence,
        "label": label,
        "formula": "0.4*retrieval + 0.3*reranker + 0.2*citation + 0.1*reasoning",
        "terms": {
            "retrieval": "Top score from Phase 9/10 retrieval list",
            "reranker": "Top score from results_with_global_rerank (Phase 10)",
            "citation": "Grounding strength from Phase 11 citations + validation",
            "reasoning": "Phase 11 structured reasoning uncertainty mapped to a score",
        },
    }


# -----------------------------------------------------------------------------
# Phase 13
# -----------------------------------------------------------------------------

def validate_phase13(record: Dict[str, Any], phase12: Dict[str, Any]) -> Dict[str, Any]:
    """Run final strict checks before showing the answer to the user."""
    phase11 = _phase11_block(record)

    citations = _dedupe_keep_order([_norm_text(c) for c in _as_list(phase11.get("citations"))])
    validation = _as_dict(phase11.get("validation"))
    sr = _as_dict(phase11.get("structured_reasoning"))

    summary_answer = _norm_text(phase11.get("summary_answer"))
    detailed_answer = _norm_text(phase11.get("detailed_answer"))
    final_answer = _norm_text(phase11.get("final_answer"))

    completeness = _norm_text(validation.get("completeness")).lower() or "partial"
    answer_verified = bool(validation.get("answer_verified", False))
    confidence = _to_float(phase12.get("score"), 0.0)
    uncertainty = _norm_text(sr.get("uncertainty")).lower() or "medium"

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

    if _as_list(phase11.get("warnings")):
        warnings.extend([_norm_text(x) for x in _as_list(phase11.get("warnings")) if _norm_text(x)])

    if uncertainty == "high":
        warnings.append("reasoning uncertainty is high")

    if issues:
        risk_level = "high"
        display_answer_type = "summary"
        answer = summary_answer or detailed_answer or final_answer or "The evidence is not strong enough to provide a reliable answer."
    elif confidence < 0.70 or completeness == "partial" or uncertainty == "medium":
        risk_level = "medium"
        display_answer_type = "detailed"
        answer = detailed_answer or final_answer or summary_answer
    else:
        risk_level = "low"
        display_answer_type = "final"
        answer = final_answer or detailed_answer or summary_answer

    answer = _norm_text(answer)

    matched_sections = _dedupe_keep_order([_norm_text(x) for x in _as_list(phase11.get("matched_sections"))])
    selected_section = _norm_text(_as_dict(phase11.get("structured_reasoning")).get("selected_section"))

    if selected_section and matched_sections and selected_section not in matched_sections:
        warnings.append("selected section is not in the matched sections list")

    return {
        "answer": answer,
        "summary_answer": summary_answer,
        "detailed_answer": detailed_answer,
        "final_answer": final_answer,
        "display_answer_type": display_answer_type,
        "citations": citations,
        "confidence": phase12,
        "risk_level": risk_level,
        "validation": {
            "answer_verified": answer_verified,
            "completeness": completeness,
            "warnings": _dedupe_keep_order(warnings),
            "verification_issues": _dedupe_keep_order(issues),
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
python validation/validate.py \
  --input_json reasoning/res_3__4.json \
  --output validation/final_3__.json \
  --pretty
  

'''