# Public Reproduction Plan

Generated: 2026-07-09T15:58:38

The public 7266.72 baseline is already reproduced locally and submitted once under the configured Kaggle user. The next reproduction target is not another full notebook run; it is task-level diff extraction against near-7266.48/7266.72 public artifacts.

Next reproduction actions:

1. Extract Ryosuke/KaggLoop artifact set and run `c_score_scan_artifacts.py --tasks P0P1 --score-top-n 20 --full-validate` if available locally.
2. Compare prvsiyan task158/task286/task054 graphs with public Python solutions and visualization notes.
3. Build one dedicated compact builder at a time, starting with task364 or task054.
