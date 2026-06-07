from __future__ import annotations

"""CLI runner for the BNS RAG evaluation benchmark.

This module is intentionally separate from the production pipeline.  It imports
LegalRAGPipeline, calls the public phase methods, and writes evaluation outputs
under backend/evaluate/results/.
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


EVALUATE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVALUATE_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

for path in (str(BACKEND_DIR), str(EVALUATE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from metrics import evaluate_record, final_answer_text  # noqa: E402
from report import write_reports  # noqa: E402


DEFAULT_GROUND_TRUTH = EVALUATE_DIR / "ground_truths" / "2_1.json"
DEFAULT_OUTPUT_DIR = EVALUATE_DIR / "results"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _safe_name(value: Any) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value or "item"))
    return "_".join(part for part in text.split("_") if part)[:80] or "item"


def _normalize_final(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    return {"value": payload}


def load_ground_truth(path: Path, query_id: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    data = _load_json(path)
    if isinstance(data, dict) and isinstance(data.get("annotations"), list):
        data = data["annotations"]
    if not isinstance(data, list):
        raise TypeError(f"Ground truth must be a list of records: {path}")
    records = [row for row in data if isinstance(row, dict)]
    if query_id:
        records = [row for row in records if str(row.get("query_id")) == query_id]
    if limit is not None:
        records = records[: max(0, limit)]
    if query_id and not records:
        raise ValueError(f"No ground-truth row found for query id {query_id}")
    return records


def run_one(
    pipeline: Any,
    record: Dict[str, Any],
    query_dir: Path,
    k: int,
) -> Dict[str, Any]:
    query = str(record.get("query") or "").strip()
    query_id = str(record.get("query_id") or "unknown")
    timings: Dict[str, float] = {}
    query_dir.mkdir(parents=True, exist_ok=True)
    _save_json(query_dir / "ground_truth.json", record)

    try:
        t0 = time.perf_counter()
        phase8 = pipeline.analyze(query)
        timings["phase_8"] = time.perf_counter() - t0
        _save_json(query_dir / "phase_8_analysis.json", phase8)

        t0 = time.perf_counter()
        retrieval = pipeline.retrieve(phase8)
        timings["phase_9_10"] = time.perf_counter() - t0
        _save_json(query_dir / "phase_9_10_retrieval.json", retrieval)

        t0 = time.perf_counter()
        phase11 = pipeline.reason(retrieval)
        timings["phase_11"] = time.perf_counter() - t0
        _save_json(query_dir / "phase_11_reasoning.json", phase11)

        t0 = time.perf_counter()
        final = _normalize_final(pipeline.validate(phase11))
        timings["phase_12_13"] = time.perf_counter() - t0
        _save_json(query_dir / "phase_12_13_validation.json", final)

        timings["total"] = sum(timings.values())
        _save_json(query_dir / "timings.json", timings)

        result = evaluate_record(record, retrieval, final, phase11_payload=phase11, k=k)
        result["timings"] = {key: round(value, 4) for key, value in timings.items()}
        answer = final_answer_text(final)
        result["final_answer_preview"] = answer
        result["raw_output_dir"] = str(query_dir)
        _save_json(query_dir / "evaluation.json", result)
        return result
    except Exception as exc:  # keep benchmark runs from dying on one query
        error = {
            "query_id": query_id,
            "query": query,
            "query_type": record.get("query_type"),
            "expected_behavior": record.get("expected_behavior"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "raw_output_dir": str(query_dir),
        }
        _save_json(query_dir / "error.json", error)
        return error


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BNS RAG evaluation against ground_truths/2.json")
    parser.add_argument("--ground-truth", default=str(DEFAULT_GROUND_TRUTH), help="Ground truth JSON file")
    parser.add_argument("--query-id", default=None, help="Run one query id, e.g. Q046")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N selected records")
    parser.add_argument("--k", type=int, default=5, help="Top-K for retrieval metrics")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Base output directory for eval runs")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    ground_truth = Path(args.ground_truth)
    output_base = Path(args.output_dir)
    if not ground_truth.is_absolute():
        ground_truth = PROJECT_ROOT / ground_truth
    if not output_base.is_absolute():
        output_base = PROJECT_ROOT / output_base

    records = load_ground_truth(ground_truth, query_id=args.query_id, limit=args.limit)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    _save_json(
        run_dir / "config.json",
        {
            "ground_truth": str(ground_truth),
            "query_id": args.query_id,
            "limit": args.limit,
            "k": args.k,
            "records": len(records),
        },
    )

    print(f"Loading pipeline once for {len(records)} query/queries...")
    from main import LegalRAGPipeline  # imported lazily so --help and tests stay lightweight

    pipeline = LegalRAGPipeline()
    results: List[Dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        query_id = str(record.get("query_id") or f"Q{idx:03d}")
        query_dir = run_dir / "raw" / f"{idx:03d}_{_safe_name(query_id)}"
        print(f"[{idx}/{len(records)}] {query_id}: {record.get('query')}")
        results.append(run_one(pipeline, record, query_dir=query_dir, k=args.k))

    summary = write_reports(run_dir, results)
    print(f"\nEvaluation saved to: {run_dir}")
    print(json.dumps(summary.get("overall", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
