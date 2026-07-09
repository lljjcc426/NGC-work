# Score Work Report

Generated: 2026-07-09T15:58:38

## Modes

- Mode A: ran P0/P1 artifact reuse scan and official full-validation cost scoring.
- Mode B: created reusable artifact scanner, cost diff runner, candidate validator, task card generator, and surgery probe scripts.
- Mode C: summarized local public notebook/bundle intel, centered on the verified prvsiyan 7266.72 baseline.
- Mode D: candidate register updated, but no new candidate package was built because there were zero accepted C replacements.
- Mode E: minimal score docs/task cards generated to support next experiments.

## Direct Score Attempts

- P0/P1 artifacts indexed: `144`.
- Artifact rows full-scored: `73`; accepted: `0`.
- Surgery rows full-scored: `56`; accepted: `0`.
- Generic optimizer/simplifier passes did not reduce official cost; several files became larger on disk but cost stayed identical.

## Cost Results

| source | scored_rows | accepted | best_delta_cost |
| --- | ---: | ---: | ---: |
| artifact_scan_top5 | 73 | 0 | 0 |
| onnx_surgery_probe | 56 | 0 | 0 |

## Quick-Win Top 10

| rank | task | priority | current_cost | current_points |
| ---: | --- | --- | ---: | ---: |
| 1 | task158 | P0_lt16 | 28483.0 | 14.742937 |
| 2 | task286 | P0_lt16 | 26909.0 | 14.799784 |
| 3 | task054 | P0_lt16 | 25394.0 | 14.857732 |
| 4 | task349 | P0_lt16 | 14892.0 | 15.391421 |
| 5 | task364 | P0_lt16 | 14642.0 | 15.408351 |
| 6 | task077 | P0_lt16 | 7657.0 | 16.056624 |
| 7 | task009 | P1_16_16p7 | 6694.0 | 16.191033 |
| 8 | task383 | P1_16_16p7 | 5830.0 | 16.329228 |
| 9 | task382 | P1_16_16p7 | 5695.0 | 16.352656 |
| 10 | task096 | P1_16_16p7 | 7678.0 | 16.053886 |

## Next 5 Experiments

1. Dedicated compact builder for `task158` motif-copy/fill.
2. Dedicated compact builder for `task286` repeated propagation.
3. Dedicated compact builder for `task054` marker-driven cross/line overwrite.
4. Dedicated compact component-shape classifier for `task364`.
5. Mine prvsiyan visualizations and KaggLoop 7266.48 for task-level graph differences before more generic surgery.

## Blockers

- No accepted lower-cost artifact found in local public artifact pool for C P0/P1.
- Existing current graphs are already optimizer-stable for official cost; generic simplification is not enough.
- Real score improvement now needs task-specific ONNX construction, not more template documentation.

## Git

Git checkpoint attempted after this report generation; see final assistant summary for commit status.
