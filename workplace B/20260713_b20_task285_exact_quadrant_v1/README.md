# B-20 task285 exact quadrant rewrite

This folder contains an independent, hidden-safe rewrite of B task285. The
model is derived from the exact ARC-GEN generator `task_b775ac94.py`; it is not
a public-model graft or a training-example lookup table.

## Rule reconstruction

The input contains one to three sprites. Each sprite is fully shown in one
quadrant around a conceptual 2x2 center and has colored one-cell anchors in
the other quadrants. The output reflects the complete sprite into every
anchored quadrant, using that quadrant's anchor color.

The rewrite uses fixed `TopK(33)` extraction, exact source-quadrant
resolution, a strict 5x5 source window, and six rounds of masked connected
expansion. It also handles the diagonal-start layout present in the official
examples.

## Safe compact representation

Earlier task285 candidates removed padding or changed `TopK` to `uint8`.
Those passed local checks but were either hidden-unsafe or returned Kaggle
`ERROR`. This model keeps the online-compatible float16 `TopK` path.

The main new reduction is signed int8 scalar color encoding: padded cells are
`-1`, in-grid background is `0`, and colors are `1..9`. This makes the padded
map itself the boundary sentinel and removes a separate final sentinel
construction and merge chain.

## Results

| metric | baseline | candidate | change |
| --- | ---: | ---: | ---: |
| cost | 19700 | 18189 | -1511 |
| memory | 19280 | 17920 | -1360 |
| params | 420 | 269 | -151 |
| points | 15.111626 | 15.191428 | +0.079802 |

Validation evidence:

- Competition validation: valid.
- Official examples: 265/265 exact.
- Fresh exact-generator stress: 50000/50000 exact, seed `28520260713`.
- Full accumulated submission: 400/400 valid.

The accumulated task018 + task285 package reaches `7296.229254` locally,
which is `+0.325834` over the local score of the online 7296.04 baseline. It
is archived but not submitted because it remains below the agreed `+1.0`
direct-submit threshold.

## Files

- `models/task285.onnx`: final task model.
- `scripts/rewrite_task285_exact_quadrant.py`: reproducible builder and stress test.
- `reports/bundle_summary.json`: accumulated package metrics and SHA-256.
- `reports/changed_tasks.csv`: task018/task285 score deltas.
- `reports/full_scores.json`: complete 400-task validation report.
- `submission/submission.zip`: accumulated continuation package.

## Next target

Continue B-only independent rewrites using the current 7296.04 task-level
score table. Do not spend time on task350: its stale assignment cost was 9036,
but the accepted model is already cost 428 and scores 18.940877. The next
research pass should inspect task101 and task076 structurally, while retaining
their existing hidden-safe decision logic.
