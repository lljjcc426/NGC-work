# B-only original rewrites on the new team baseline

This folder records the B-task work based on the team-provided
`submission (2).zip`. No non-B model is changed.

## Baseline

- Source archive SHA256: `346129A9E05F2A30D303EAA2D9B8F0A052CDCCB57715C3FBA12284C3061F2DA9`
- Local validation: `400/400`
- Local score: `7377.858249106621`
- B-task count: `67`
- B-task local average: `18.392367`

Reaching 7800 with non-B tasks frozen would require the B average to become
`24.692990`, very close to the per-task theoretical maximum of 25. The practical
near-term target remains verified whole-point gains while pushing low B tasks
toward 20.

## Rewrites

| Task | Method | Cost | Local gain |
| --- | --- | ---: | ---: |
| `task018` | sparse dihedral reconstruction | `24360 -> 19047` | `+0.246033` |
| `task123` | nested-square rank factorization | `1342 -> 1020` | `+0.274358` |
| `task134` | generator-proven one-axis variance | `1529 -> 1329` | `+0.140187` |
| `task285` | exact quadrant and connected-sprite rewrite | `18674 -> 18189` | `+0.026315` |
| `task350` | shared rule tensors inside one Einsum | `428 -> 400` | `+0.067659` |
| `task360` | reuse the fold matrix as the output route | `340 -> 250` | `+0.307485` |

Combined local gain: `+1.062037`.

Projected local score: `7378.920286`.

## Validation

- Every replacement passes the official local train/test/ARC-GEN set.
- `task123`: `50,000/50,000` fresh exact-generator examples.
- `task134`: `100,000/100,000` fresh exact-generator examples.
- `task285`: official set plus prior `50,000/50,000` exact-generator stress.
- `task350`: `267/267` sign-equivalent to the baseline; tensor identities are exact.
- `task360`: `266/266` sign-equivalent to the baseline; route identity is exact.
- Final ZIP: 400 root-level ONNX files.
- Final ZIP SHA256: `77B2E974E84EE0212CD22F7114161F7394BBAA70B5F7452CD465E05BBC99DE8B`.

## Kaggle

- Submission ref: `54647712`
- Final status: `COMPLETE`
- Public score: `7379.07`
- The displayed gain over the nearby `7378.40` team submission is about
  `+0.67`. This closely matches the guaranteed task123/task285/task350/task360
  local subtotal; task018 and task134 are therefore not counted as reliable
  unpublished gains in later batches.

## Contents

- `models/`: six B-task overrides.
- `scripts/`: reproducible rewrite scripts.
- `reports/`: base score report and exact gain summary.
- `submission/base_submission.zip`: unmodified provided baseline.
- `submission/submission.zip`: complete 400-task candidate.

## Next direction

Continue original B-only structure work. The best targets remain `task018`,
`task285`, `task101`, `task076`, `task209`, and `task023`. Prefer exact generator
constraints, shared constants in single-Einsum models, and removal of full-size
intermediates. Do not repeat sparse-initializer Conv/Gather probes or task344
rank-5 CP approximation; both are rejected by checker/runtime or official data.
