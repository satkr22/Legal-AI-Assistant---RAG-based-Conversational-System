# BNS RAG Evaluation

This folder contains a standalone evaluator for the BNS legal RAG pipeline. It
uses `ground_truths/2.json` as the canonical 100-query benchmark and writes all
evaluation artifacts under `backend/evaluate/results/`.

The evaluator does not modify the existing Phase 8-13 pipeline. It imports
`LegalRAGPipeline`, calls the phase methods, and scores the outputs.

## Metrics

Retrieval:
- `Precision@5`
- `Recall@5`
- `MRR@5`
- `Support Precision@5`
- `Any Correct Support@5`

`Precision@5`, `Recall@5`, and `MRR@5` are strict required-section metrics.
`acceptable_references` are neutral and do not count as required hits. The
support metrics show whether the retriever found either a required or acceptable
section, which is useful for scenario queries where multiple provisions may be
legally relevant.

Answer quality:
- `Required Point Coverage`
- `Forbidden Claim Rate`
- `Answer Completeness Score`

Legal grounding:
- `Citation Recall`
- `Citation Precision`
- `Faithfulness`

Safety:
- `Hallucination Rate`
- `Out-of-Scope Handling Accuracy`
- `Unsafe Behavior Failure Rate`

UX / demo:
- `Confidence Score`
- `Risk Level`
- `High-Confidence Wrong Rate`
- `Low-Risk Wrong Rate`

## Commands

Run one smoke-test query:

```bash
python backend/evaluate/eval_runner.py --query-id Q046
```

Run the first five queries:

```bash
python backend/evaluate/eval_runner.py --limit 5
```

Run the corrected-row checks:

```bash
python backend/evaluate/eval_runner.py --query-id Q048
python backend/evaluate/eval_runner.py --query-id Q054
python backend/evaluate/eval_runner.py --query-id Q061
python backend/evaluate/eval_runner.py --query-id Q066
python backend/evaluate/eval_runner.py --query-id Q100
```

Run the full 100-query benchmark:

```bash
python backend/evaluate/eval_runner.py
```

Use a different top-K:

```bash
python backend/evaluate/eval_runner.py --k 10
```

## Outputs

Each run creates:

```text
backend/evaluate/results/run_<timestamp>/
  config.json
  summary.json
  per_query_results.json
  failures.md
  raw/
    001_Q001/
      ground_truth.json
      phase_8_analysis.json
      phase_9_10_retrieval.json
      phase_11_reasoning.json
      phase_12_13_validation.json
      timings.json
      evaluation.json
```

If a query fails, the runner writes `error.json` for that query and continues.

## Notes

- Scoring is section-level, not chunk-id-level.
- `acceptable_references` are treated as correct support but not required.
- `wrong_references` are flagged if retrieved or cited.
- Required/forbidden answer-point matching is deterministic and approximate;
  use `failures.md` for human review of borderline cases.
