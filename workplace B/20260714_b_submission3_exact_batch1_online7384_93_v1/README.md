# B-only exact batch on team submission 3

This folder records the first B-task batch rebased onto the team-provided
`submission (3).zip`. Only B-assigned tasks are changed.

## Baseline

- Source ZIP SHA256:
  `A5C0B4BF18F1C89A27452888FF309B9293026DE9D5441497FF8CCCDB0E9A458D`.
- Local validation: `400/400`.
- Local score: `7383.777544448324`.
- Kaggle baseline ref: `54673685`.
- Kaggle baseline public score: `7383.93`.

## Rewrites

| Task | Method | Cost | Local gain |
| --- | --- | ---: | ---: |
| `task076` | uint8 coordinate geometry; float16 TopK retained | `12825 -> 12313` | `+0.040741` |
| `task101` | precomposed flat sprite-placement offsets | `13711 -> 13071` | `+0.047802` |
| `task123` | nested-square rank factorization | `1342 -> 1020` | `+0.274358` |
| `task163` | shared terminal-Einsum lookup tables | `310 -> 298` | `+0.039479` |
| `task208` | compact search plus direct ten-channel color ArgMin | `4181 -> 4043` | `+0.033563` |
| `task209` | factorized terminal color tensor | `7604 -> 7324` | `+0.037518` |
| `task270` | factorized spatial selectors and color rules | `2846 -> 2719` | `+0.045650` |
| `task328` | dynamic-size Voronoi core with direct padding | `5744 -> 5189` | `+0.101615` |
| `task350` | shared relations plus S-diagonal output signs | `428 -> 398` | `+0.072671` |
| `task360` | reuse the fold matrix as output route | `340 -> 250` | `+0.307485` |

Combined local gain: `+1.000882707997`.

Candidate local score: `7384.778427156321`.

## Validation

- Every replacement matches the new baseline on its complete official
  train/test/ARC-GEN set.
- Full continuation package: `400/400` valid tasks, zero invalid models.
- `task208` has a minimum background-versus-frame count margin of `+26` on
  the official set. An additional `20,000` exact ARC-GEN samples had minimum
  margin `+17` and zero failures.
- `task350` S-diagonal sign sharing matches `267/267` official examples.
- Submission ZIP SHA256:
  `8EBA05F6379D98C2F99E56760468521D912D23FD0A247556BCB1DC2EF2211207`.

## Kaggle

- Submission ref: `54686944`.
- Status: `COMPLETE`.
- Public score: `7384.93`.
- Verified online gain over the same-package baseline: `+1.00`.

## Rejected experiments

- `task163` global three-state RL truncation failed on the first training
  example; its fourth relation state is still required.
- `task350` outer-G and full gate-removal variants failed on the first training
  example. Only replacing `G` with the diagonal of `S` is retained.
- The old uint8-TopK task076 model remains excluded because it failed online;
  the accepted model quantizes only bounded coordinate geometry.

## Contents

- `models/`: ten accepted B-task overrides.
- `scripts/`: reproducible rewrites and rejected-variant search scripts.
- `reports/`: baseline/candidate 400-task scores and online result summary.
- `submission/base_submission.zip`: unmodified team-provided baseline.
- `submission/submission.zip`: online-verified 400-model package.

## Resume point

Use the `7384.93` package as the next parent. Continue one B task at a time,
starting with the lowest-score tasks whose exact generator structure can remove
whole activation maps. Do not reapply these ten overrides or the rejected
task163/task350 variants.
