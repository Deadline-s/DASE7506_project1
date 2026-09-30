# Resource check (validation only)

Candidate: GELU + dropout 0.05, width 192, depth 6, 6000 steps.

CPU FP32, 4 threads; three sequential fresh processes per model, alternating baseline/candidate. Scoring time uses the unchanged evaluator's seconds field, excluding loading. Peak RSS uses macOS getrusage in bytes, includes loading and scoring. No Python process appeared in the post-measurement process check; this is not continuous monitoring.

| Measure | Result | Limit |
|---|---:|---:|
| Baseline median validation scoring | 4.249 s | Reference |
| Candidate median validation scoring | 7.077 s | 5x baseline |
| Time ratio | 1.665x | 5x |
| Candidate maximum process peak RSS | 1.711 GiB | 4 GiB |
| Checkpoint | 11.896 MiB | Part of assets |
| Conservative asset inventory | 24.796 MiB | 64 MiB |

Inventory includes checkpoint, inference modules, scorer, requirements, configuration, tokenizer and all supplied dataset splits; excludes installed Python/dependencies. It is an enumerated file set, not a packaged archive. See summary.json for exact paths and hashes.

All three checks pass on validation. This is not full-test certification; repeat measurement on the same frozen predictor during final full-test evaluation. No test scores generated in this check.
