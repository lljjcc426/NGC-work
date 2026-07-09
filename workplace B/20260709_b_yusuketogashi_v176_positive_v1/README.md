# B-only yusuketogashi V176 positive graft

Date: 2026-07-09

This folder contains one B-task-only submission candidate. It starts from our local `aggr7250_25_jack_direct_v1` baseline and grafts only the B-owned tasks where yusuketogashi's public Kaggle baseline V176 is locally better.

## Scope

- Owner scope: `workplace B` only.
- Task source: `assignments/task_assignment_400.csv`.
- B task count checked: 67.
- Non-B tasks touched: 0.
- Positive replacements kept: 14.

## Source

- Kaggle notebook: `yusuketogashi/2026-neurogolf-baseline`
- Extracted artifact: embedded V176 `submission.zip`
- Extracted zip sha256: `ae01c3182021010ca3197447c80055127b7051124cb99ab6124dc2cd370dd076`
- Built candidate sha256: `570180b360bf5acea00d08abc556423c327319cae71a980dbb6dd3d450381a47`

## Method

1. Parsed the 67 tasks assigned to B.
2. Validated yusuketogashi V176 ONNX files with our local scorer.
3. Compared every B task against `aggr7250_25_jack_direct_v1`.
4. Kept only replacements with positive local point gain.
5. Built a full 400-task `submission.zip`, changing only the 14 positive B tasks.

## Local Result

- Local base total: `7250.128424`
- Expected local gain: `+1.336384`
- Expected local total: `7251.464808`

## Kaggle Verification

- Submission ref: `54488042`
- Public score: `7251.58`
- Status: positive versus the old local 7250.25 base, but not competitive with the later 2026-07-09 public notebook baseline score `7267.31`.

| task | local gain | cost change |
| --- | ---: | ---: |
| task368 | +0.354450 | 5130 -> 3599 |
| task280 | +0.249303 | 6032 -> 4701 |
| task170 | +0.175633 | 2533 -> 2125 |
| task161 | +0.164370 | 2131 -> 1808 |
| task008 | +0.152535 | 3195 -> 2743 |
| task277 | +0.055226 | 3741 -> 3540 |
| task328 | +0.050571 | 6631 -> 6304 |
| task245 | +0.043971 | 2743 -> 2625 |
| task270 | +0.040122 | 2975 -> 2858 |
| task131 | +0.025989 | 3976 -> 3874 |
| task247 | +0.017205 | 469 -> 461 |
| task076 | +0.004745 | 12886 -> 12825 |
| task090 | +0.001309 | 3058 -> 3054 |
| task208 | +0.000955 | 4189 -> 4185 |

## Files

- `submission.zip`: full 400-task candidate for Kaggle submission; only the 14 B tasks above differ from the local base.
- `overrides/`: the 14 replacement ONNX files.
- `summary.json`: compact build summary.
- `manifest.json`: source, scope, hash, and expected score metadata.
- `reports/b_positive_replacements.csv`: positive B-task replacements only.
- `reports/b_all_67_yusuke_compare.csv`: all 67 B-task comparisons.
- `reports/b_task_assignments.json`: parsed B assignment list.
- `reports/b_yusuketogashi_vs_aggr7250_25_scores.json`: detailed scorer output.

## Risk Notes

- This is a donor-ONNX graft from a real public Kaggle notebook, not a train-example lookup rewrite.
- We previously saw a hidden-score failure from an exact lookup style experiment, so LB verification is still required before treating this as final.
- Our current online best has moved above this candidate, so this folder should be treated as a B-task evidence package and replacement source, not as the final best submission.

## Next B-only Targets

After this graft, the next B path should focus on high-cost B tasks that remain weak and are less likely to be solved by copy-blending alone:

- P0: `task018`, `task285`, `task101`, `task350`, `task255`, `task023`.
- P1: `task209`, `task205`, `task377`, `task212`, `task228`, `task344`.

Recommended direction: write hidden-safe rule-equivalent ONNX for these tasks, not exact training-output tables.
