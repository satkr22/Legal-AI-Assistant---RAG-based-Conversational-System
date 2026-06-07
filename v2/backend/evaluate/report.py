from __future__ import annotations

"""Report writers for BNS RAG evaluation runs."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from metrics import average


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _metric(result: Dict[str, Any], *keys: str) -> Optional[float]:
    cur: Any = result
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur if isinstance(cur, (int, float)) else None


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed = [r for r in results if not r.get("error")]
    failed = [r for r in results if r.get("error")]

    def section(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "count": len(rows),
            "answer_accuracy": average(1.0 if r.get("answer_correct") else 0.0 for r in rows),
            "retrieval_precision_at_k": average(_metric(r, "retrieval", "precision_at_k") for r in rows),
            "retrieval_recall_at_k": average(_metric(r, "retrieval", "recall_at_k") for r in rows),
            "retrieval_mrr_at_k": average(_metric(r, "retrieval", "mrr_at_k") for r in rows),
            "retrieval_support_precision_at_k": average(
                _metric(r, "retrieval", "support_precision_at_k") for r in rows
            ),
            "retrieval_any_correct_support_at_k": average(
                1.0 if (r.get("retrieval") or {}).get("any_correct_support_at_k") else 0.0 for r in rows
            ),
            "candidate_precision_at_10": average(_metric(r, "candidate_retrieval", "precision_at_k") for r in rows),
            "candidate_recall_at_10": average(_metric(r, "candidate_retrieval", "recall_at_k") for r in rows),
            "candidate_mrr_at_10": average(_metric(r, "candidate_retrieval", "mrr_at_k") for r in rows),
            "candidate_any_correct_support_at_10": average(
                1.0 if (r.get("candidate_retrieval") or {}).get("any_correct_support_at_k") else 0.0
                for r in rows
            ),
            "prompt_precision_at_5": average(_metric(r, "prompt_evidence", "precision_at_k") for r in rows),
            "prompt_recall_at_5": average(_metric(r, "prompt_evidence", "recall_at_k") for r in rows),
            "prompt_mrr_at_5": average(_metric(r, "prompt_evidence", "mrr_at_k") for r in rows),
            "prompt_any_correct_support_at_5": average(
                1.0 if (r.get("prompt_evidence") or {}).get("any_correct_support_at_k") else 0.0 for r in rows
            ),
            "selector_loss_rate": average(
                1.0 if (r.get("prompt_evidence") or {}).get("selector_loss") else 0.0 for r in rows
            ),
            "soft_answer_score": average(_metric(r, "answer_quality", "soft_answer_score") for r in rows),
            "required_point_coverage": average(_metric(r, "answer_quality", "required_point_coverage") for r in rows),
            "forbidden_claim_rate": average(_metric(r, "answer_quality", "forbidden_claim_rate") for r in rows),
            "forbidden_contradiction_skipped_rate": average(
                _metric(r, "answer_quality", "forbidden_contradiction_skipped_rate") for r in rows
            ),
            "citation_recall": average(_metric(r, "grounding", "citation_recall") for r in rows),
            "citation_precision": average(_metric(r, "grounding", "citation_precision") for r in rows),
            "faithfulness": average(_metric(r, "grounding", "faithfulness") for r in rows),
            "hallucination_rate": average(_metric(r, "safety", "hallucination_rate") for r in rows),
            "confidence_score": average(_metric(r, "ux", "confidence_score") for r in rows),
            "expected_behavior_accuracy": average(
                1.0 if (r.get("safety") or {}).get("expected_behavior_ok") else 0.0 for r in rows
            ),
        }

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in completed:
        by_type[str(row.get("query_type") or "unknown")].append(row)

    out_of_scope = [r for r in completed if r.get("query_type") == "out_of_scope"]
    adversarial = [r for r in completed if r.get("query_type") == "adversarial"]
    high_conf_wrong = [r for r in completed if (r.get("ux") or {}).get("high_confidence_wrong")]
    low_risk_wrong = [r for r in completed if (r.get("ux") or {}).get("low_risk_wrong")]

    failure_counter: Counter[str] = Counter()
    for row in completed:
        failure_counter.update(row.get("failure_reasons") or [])

    return {
        "total": len(results),
        "completed": len(completed),
        "pipeline_errors": len(failed),
        "overall": section(completed),
        "by_query_type": {key: section(rows) for key, rows in sorted(by_type.items())},
        "safety": {
            "out_of_scope_handling_accuracy": average(
                1.0 if (r.get("safety") or {}).get("out_of_scope_handling_ok") else 0.0
                for r in out_of_scope
            ),
            "unsafe_behavior_failure_rate": average(
                0.0 if (r.get("safety") or {}).get("unsafe_behavior_ok") else 1.0
                for r in adversarial
            ),
            "out_of_scope_count": len(out_of_scope),
            "adversarial_count": len(adversarial),
        },
        "ux": {
            "high_confidence_wrong_count": len(high_conf_wrong),
            "low_risk_wrong_count": len(low_risk_wrong),
            "risk_level_counts": dict(Counter((r.get("ux") or {}).get("risk_level", "unknown") for r in completed)),
        },
        "failure_reason_counts": dict(failure_counter),
        "worst_failures": [
            {
                "query_id": r.get("query_id"),
                "query": r.get("query"),
                "query_type": r.get("query_type"),
                "failure_reasons": r.get("failure_reasons"),
                "confidence_score": (r.get("ux") or {}).get("confidence_score"),
                "risk_level": (r.get("ux") or {}).get("risk_level"),
            }
            for r in completed
            if r.get("failure_reasons")
        ][:25],
        "errors": failed,
    }


def write_failures_markdown(path: Path, results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    lines = [
        "# BNS RAG Evaluation Failures",
        "",
        f"Total queries: {summary['total']}",
        f"Completed: {summary['completed']}",
        f"Pipeline errors: {summary['pipeline_errors']}",
        f"High-confidence wrong: {summary['ux']['high_confidence_wrong_count']}",
        f"Low-risk wrong: {summary['ux']['low_risk_wrong_count']}",
        "",
        "## Failure Reason Counts",
        "",
    ]
    if summary["failure_reason_counts"]:
        for reason, count in sorted(summary["failure_reason_counts"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- No deterministic failures detected.")

    lines.extend(["", "## Query Failures", ""])
    failures = [r for r in results if r.get("failure_reasons") or r.get("error")]
    for row in failures:
        lines.append(f"### {row.get('query_id')} - {row.get('query_type')}")
        lines.append("")
        lines.append(f"Query: {row.get('query')}")
        if row.get("error"):
            lines.append(f"Error: `{row.get('error')}`")
            lines.append("")
            continue
        lines.append(f"Failures: {', '.join(row.get('failure_reasons') or [])}")
        lines.append(f"Required sections: {', '.join(row.get('required_sections') or []) or '-'}")
        lines.append(f"Retrieved@K: {', '.join(row.get('retrieved_sections_at_k') or []) or '-'}")
        lines.append(f"Candidate@10: {', '.join(row.get('candidate_sections_at_10') or []) or '-'}")
        lines.append(f"Prompt@5: {', '.join(row.get('prompt_evidence_sections_at_5') or []) or '-'}")
        lines.append(f"Cited: {', '.join(row.get('cited_sections') or []) or '-'}")
        lines.append(
            "Scores: "
            f"R@K={((row.get('retrieval') or {}).get('recall_at_k'))}, "
            f"MRR={((row.get('retrieval') or {}).get('mrr_at_k'))}, "
            f"SupportHit={((row.get('retrieval') or {}).get('any_correct_support_at_k'))}, "
            f"PromptR@5={((row.get('prompt_evidence') or {}).get('recall_at_k'))}, "
            f"SelectorLoss={((row.get('prompt_evidence') or {}).get('selector_loss'))}, "
            f"CitationRecall={((row.get('grounding') or {}).get('citation_recall'))}, "
            f"PointCoverage={((row.get('answer_quality') or {}).get('required_point_coverage'))}, "
            f"SoftScore={((row.get('answer_quality') or {}).get('soft_answer_score'))}"
        )
        ux = row.get("ux") or {}
        lines.append(f"Confidence/Risk: {ux.get('confidence_score')} / {ux.get('risk_level')}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reports(output_dir: Path, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = summarize(results)
    _save_json(output_dir / "per_query_results.json", results)
    _save_json(output_dir / "summary.json", summary)
    write_failures_markdown(output_dir / "failures.md", results, summary)
    return summary
