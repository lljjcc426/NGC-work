# B-only exact rewrite batch 2

This folder continues from the team-provided `submission (2).zip` and the
accepted B-only parent submission. Only B-assigned tasks are changed.

## Parent

- Original team baseline local score: `7377.858249106621`.
- Accepted parent Kaggle ref: `54647712`.
- Accepted parent public score: `7379.07`.
- Parent package local score: `7378.920286013601`.

## Exact rewrites

| Task | Method | Cost | Gain |
| --- | --- | ---: | ---: |
| `task076` | uint8 coordinate and geometry path; float16 TopK retained | `12825 -> 12313` | `+0.040741` |
| `task101` | precomposed flat sprite-placement offsets | `13711 -> 13071` | `+0.047802` |
| `task163` | shared terminal-Einsum lookup tables | `310 -> 298` | `+0.039479` |
| `task208` | compact color selection, outer kernel, ScatterND index, and OneHot | `4181 -> 4084` | `+0.023474` |
| `task209` | factorized terminal color tensor | `7609 -> 7324` | `+0.038175` |
| `task270` | factorized spatial selectors and color rules | `2846 -> 2719` | `+0.045650` |
| `task328` | dynamic-size Voronoi core with direct output padding | `5746 -> 5189` | `+0.101963` |

Combined unpublished gain: `+0.337283958434`.

Full candidate local score: `7379.257569972036`.

Projected online score from the accepted `7379.07` parent: approximately
`7379.41`. No Kaggle submission was made because this batch is below the
agreed `+1.0` direct-submit threshold.

## Validation

- Full continuation package: `400/400` valid tasks, zero invalid models.
- `task076`, `task101`, `task163`, `task208`, `task209`, and `task270` match
  every official train/test/ARC-GEN example for their tasks.
- `task328` matches all `267/267` official examples. Its dynamic dimensions
  carry static maximum-size annotations because the competition scorer rejects
  symbolic intermediate shapes; ONNX Runtime and the official scorer both pass.
- Submission ZIP SHA256:
  `916737473FCBE40E2AD0C8176092ACFEC6F6EC287E7B0F3A122F3D09CC273CEF`.

## Contents

- `models/`: seven exact B-task overrides.
- `scripts/`: reproducible rewrite scripts plus the Einsum factor scanner.
- `reports/changed_tasks.csv`: per-task costs, gains, and model hashes.
- `reports/summary.json`: machine-readable batch summary.
- `reports/full_scores.json`: full 400-task validation and score report.
- `submission/submission.zip`: root-level 400-model continuation package.

## Resume point

The next run should continue from this package and keep the current exact-rule
approach. The reliable unpublished gain is `+0.337284`; accumulate at least
another `+0.662716` before the next direct Kaggle verification. Do not use the
older uint8-TopK task076 experiment: it failed online. The retained task076
model keeps TopK on float16 and quantizes only bounded coordinate geometry.
