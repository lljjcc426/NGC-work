# B-only yusuke V176 task023 + task255 improvement

Date: 2026-07-09

This folder contains the current B-only improvement package on top of the stronger yusuketogashi V176 public baseline.

## Baseline

- Source baseline: `yusuketogashi/2026-neurogolf-baseline` embedded V176 submission.
- Kaggle public baseline score observed: `7267.31`.
- Local scorer baseline total: `7267.145162`.
- Scope: B tasks only. Non-B tasks are unchanged.

## Changes

| task | source | local gain | cost change |
| --- | --- | ---: | ---: |
| task023 | historical `aggr7162_ext7000_t023_merge` override | +0.327464 | 8222 -> 5926 |
| task255 | new QLinearMatMul tail rewrite | +0.168126 | 8911 -> 7532 |

Combined local gain: `+0.495590`.

Expected local total: `7267.640752`.

This is below the previous auto-submit threshold of `+1.0`, so it is recorded as an accumulating candidate rather than submitted immediately.

## Method

1. Re-based B work on yusuketogashi V176 instead of the older 7250.25 local baseline.
2. Scored 67 B tasks under the local scorer and sorted low-score/high-cost targets.
3. Re-ran historical B task-level candidates against the new yusuke baseline.
4. Found that almost all older B overrides were worse than yusuke V176, except `task023`.
5. Added a new `task255` graph rewrite: replace the final float MatMul mask test with `QLinearMatMul` over uint8 0/1 tensors, then compare against uint8 zero.

## Files

- `submission.zip`: full 400-task candidate based on yusuke V176, changing only `task023` and `task255`.
- `overrides/task023.onnx`: the positive historical task023 override.
- `overrides/task255.onnx`: the new QLinearMatMul task255 rewrite.
- `summary.json`: compact score and hash summary.
- `reports/`: score evidence for selected public variants, named single-task candidates, and task255 rewrite.
- `scripts/optimize_task255_u8_matmul_tail_yusuke.py`: accepted task255 rewrite script.
- `scripts/score_task_candidate_pool_robust.py`: robust per-candidate scorer used to avoid broken candidate models killing the whole pool.
- `scripts/rejected/optimize_task023_label_equal_tail_yusuke.py`: rejected task023 tail rewrite. It was explored but did not pass full validation across examples.

## Current Status

- Current accumulated B gain over yusuke V176: `+0.495590`.
- Need roughly another `+0.5` local gain before this meets the direct-submit threshold.
- Best next targets remain `task018`, `task285`, `task101`, `task076`, `task350`, `task209`, and `task328`, but old public grafts did not beat yusuke on these tasks.

## Notes

- The earlier B-only yusuketogashi graft scored `7251.58` online and is now only evidence against the old 7250.25 base.
- The active baseline for new B work is yusuke V176 / public `7267.31`.
- Avoid train-example lookup rewrites; prioritize hidden-safe graph rewrites and verified task-level candidates.
