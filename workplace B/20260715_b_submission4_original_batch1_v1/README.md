# B-only original rewrite batch on team submission 4

This folder records ten independently derived B-task rewrites on the
team-provided `submission (4).zip`. No public submission package was blended
into this batch, and no non-B task was changed.

## Baseline

- Source ZIP SHA256:
  `03849373E14E07E2AE3D5664A4BEA4144EF4F3CC137E1E2E1B6EE71DC6509403`.
- Local validation: `400/400`.
- Local score: `7386.993113263445`.

## Rewrites

| Task | Method | Cost | Local gain |
| --- | --- | ---: | ---: |
| `task018` | compact target-domain propagation and selection | `24360 -> 16679` | `+0.378792` |
| `task023` | direct gray-minus-boxes label construction | `6353 -> 6217` | `+0.021640` |
| `task076` | even boundary sentinel with compact safe convolution path | `12313 -> 11832` | `+0.039848` |
| `task101` | binary TopK source extraction and placement | `13071 -> 11352` | `+0.141002` |
| `task209` | compact generator-aligned core | `7324 -> 6374` | `+0.138929` |
| `task270` | prefix-sum directional selectors | `2719 -> 2632` | `+0.032520` |
| `task280` | int8 coordinate geometry | `4588 -> 4335` | `+0.056723` |
| `task285` | compact legacy diagonal routing | `18189 -> 17025` | `+0.066134` |
| `task328` | compact dynamic tail | `5189 -> 5153` | `+0.006962` |
| `task344` | jointly trained rank-2 spatial basis with hard-negative refinement | `692 -> 598` | `+0.145995` |

Combined local gain: `+1.028545100559`.

Candidate local score: `7388.021658364004`.

## Task344 Method

The original single-Einsum model uses a `10 x 10 x 3` spatial tensor. A plain
rank-2 SVD failed because its third slice carries boundary/background state.
The accepted rewrite jointly optimizes the rank-2 spatial basis and the
existing `U/G/A` color factors against all official examples. Average margin
training reduced the residual to 15 bits; full-dataset top-k hard-negative
refinement then reached zero errors. The final ONNX remains one Einsum, has no
scored intermediate memory, and cuts parameters from 692 to 598.

## Validation

- Every changed task passes its complete train, test, and ARC-GEN set.
- The merged directory contains exactly 400 models.
- The ZIP contains `task001.onnx` through `task400.onnx` at archive root.
- Submission ZIP SHA256:
  `738E1CFC91A4BB9306E58D133BC6C30148964E7026B10EC3C74EBB6DC14FF7AB`.

## Kaggle Status

The package crossed the direct-submit threshold, but no Kaggle submission was
created from this runtime. Authentication and submission-list APIs work; the
signed upload endpoint is hosted by `www.googleapis.com`, which is unreachable
from the current environment and times out before any bytes are transferred.
Therefore `7388.021658` is locally verified, not an online score.

## Contents

- `models/`: ten accepted B-task overrides.
- `scripts/`: reproducible rewrites plus task344 training and diagnostics.
- `reports/baseline_full_scores.json`: official local score of all baseline tasks.
- `reports/summary.json`: exact per-task gains and package hashes.
- `submission/submission.zip`: the complete 400-model verification package.

## Resume Point

Resume from this exact package for further work on the same provided baseline.
Do not reapply these ten overrides. The repository has newer whole-team Kaggle
submissions, but they are intentionally not blended here because this round was
requested as an isolated B-only rewrite pass.
