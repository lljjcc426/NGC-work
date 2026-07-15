# B-only original rewrite batch on team submission 4

This folder records nine accepted and one rejected independently derived B-task rewrite on the
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
Combined accepted local gain: `+0.882549898892`.

Safe candidate local score: `7387.875663162337`.

## Task344 Method

The original single-Einsum model uses a `10 x 10 x 3` spatial tensor. A plain
rank-2 SVD failed because its third slice carries boundary/background state.
The rejected rewrite jointly optimizes the rank-2 spatial basis and the
existing `U/G/A` color factors against all official examples. Average margin
training reduced the residual to 15 bits; full-dataset top-k hard-negative
refinement then reached zero local errors. It failed hidden generated examples
at Kaggle ref `54732385`, zeroing all task344 points. The baseline task344 is
restored in the safe package; the failed model is retained under `rejected/`.

## Validation

- Every accepted changed task passes its complete train, test, and ARC-GEN set.
- The merged directory contains exactly 400 models.
- The ZIP contains `task001.onnx` through `task400.onnx` at archive root.
- Safe9 ZIP SHA256:
  `914ABBF367A25EF62120C9E8408558D5E4AA56E3C051F710A217B329770B557F`.

## Kaggle Status

The ten-task package was submitted as ref `54732385` and completed at
`7369.57`. The exact score identity
`7387.15 + 0.882549899 - 18.460414044 = 7369.5721` isolates task344 as the
hidden failure. The remaining nine-task package is locally verified but remains
below the next `+1.0` direct-submit threshold by `0.117450101`.

## Contents

- `models/`: nine accepted B-task overrides.
- `rejected/`: the hidden-unsafe task344 rank-2 model.
- `scripts/`: reproducible rewrites plus task344 training and diagnostics.
- `reports/baseline_full_scores.json`: official local score of all baseline tasks.
- `reports/summary.json`: exact per-task gains and package hashes.
- `submission/submission_online_failed_54732385.zip`: the failed online package.
- `submission/submission_safe9_unsubmitted.zip`: task344 restored to baseline.

## Resume Point

Resume from `submission_safe9_unsubmitted.zip` for further work on the same
provided baseline. Do not reapply the nine accepted overrides or task344 rank-2
rewrite. The repository has newer whole-team Kaggle
submissions, but they are intentionally not blended here because this round was
requested as an isolated B-only rewrite pass.
