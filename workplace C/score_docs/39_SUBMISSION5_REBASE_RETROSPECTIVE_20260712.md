# Submission 5 Rebase Retrospective - 2026-07-12

## Result

| field | value |
| --- | --- |
| parent file | `E:/submission (5).zip` |
| parent SHA256 | `d53db8c5eb5111d065f7fcc241581584da0930a06fa9c15145364bae1c14e47b` |
| parent Kaggle ref | `54606556` |
| parent public score | `7277.83` |
| experiment | `GOLF_20260712_102_submission5_plus_c22` |
| submission ref | `54608730` |
| public score | `7278.75` |
| observed gain | `+0.92` |
| expected local gain | `+0.9191029246708331` |
| status | complete, local expectation confirmed |

## Rebase Method

The user-provided parent contained exactly 400 ONNX files. The C experiment
ledger was filtered to rows with `accepted=true`, a numeric new cost, and an
existing artifact. For every task, only the lowest-cost accepted artifact was
retained. The parent and candidate were then independently scored with the
official local utility over all public train/test/arc-gen examples.

An artifact was overlaid only when both models were valid and
`candidate_cost < parent_cost`. Equal-cost replacements were deliberately
skipped. This prevented the 7277.83 parent from being overwritten by stale or
weaker C candidates.

## Selected Replacements

`task009, task046, task072, task077, task091, task094, task096, task132,
task158, task165, task190, task224, task237, task332, task349, task364,
task378, task381, task382, task383, task388, task392`

The largest parent-relative cost reductions were:

| task | parent cost | candidate cost | saved cost |
| --- | ---: | ---: | ---: |
| task158 | 28483 | 26250 | 2233 |
| task349 | 14647 | 12480 | 2167 |
| task096 | 7678 | 6850 | 828 |
| task077 | 7657 | 7234 | 423 |
| task332 | 561 | 438 | 123 |
| task237 | 1836 | 1716 | 120 |
| task009 | 6694 | 6585 | 109 |
| task072 | 421 | 368 | 53 |

## Skipped Equal-Cost Tasks

The parent already contained the current C best for `task069`, `task193`,
`task201`, `task230`, `task286`, `task298`, `task335`, and `task372`. These
were not rewritten because their cost delta was zero.

## Build And Submission

1. The first build was stopped by the evidence gate because the new experiment
   ID was not listed in the existing ONNX-surgery direction.
2. The supplied direction helper was verified to be read-only, so the
   experiment ID was added explicitly to the direction registry.
3. The next build exposed an override schema mismatch: the workflow requires
   `candidate_model_path`, not `model_path`.
4. After correcting the generated CSV header, the 400-file candidate passed
   validation, was packed, and produced a Kaggle notebook.
5. One kernel/submission run was made. No automatic retry was used.

## Conclusion

Parent-aware cost comparison was essential. Blindly stacking every historical
C artifact would have added eight no-op replacements and could have hidden
stale regressions. The filtered 22-task overlay reproduced the local score
estimate to leaderboard precision and established a new team score of
`7278.75`.
