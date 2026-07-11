# B-only online-safe v5

This folder contains the verified B-task patch set produced by direct ONNX
rewrites. No non-B task was changed.

## Result

- Original baseline: yusuketogashi V176
- Original local total: `7267.145162`
- Local gain: `+1.030562`
- Expected local total: `7268.175724`
- Kaggle ref: `54568534`
- Kaggle public score: `7268.31`
- Online gain over the V176 public score `7267.31`: `+1.00`
- Zip SHA256: `8b11b387509b9ae60cbd927ef71695180cf2f70105b67df0a877742de18402d7`

The team had a newer non-B aggregate at `7271.93` while this work was being
verified. Treat this folder as a verified B-task overlay for that newer base,
not as a replacement for the team's full aggregate.

## Included tasks

| task | rewrite | cost | local gain |
| --- | --- | ---: | ---: |
| `task205` | float16 output tail | `4891 -> 4251` | `+0.140243` |
| `task209` | one template pad after `Concat(template, zero, template)` | `7775 -> 7631` | `+0.018695` |
| `task255` | uint8 `QLinearMatMul` tail | `8911 -> 7532` | `+0.168126` |
| `task277` | remove one redundant A/B MaxPool iteration | `3540 -> 3140` | `+0.119904` |
| `task328` | algebraic corner-distance split, removing two full distance maps | `6304 -> 5746` | `+0.092680` |
| `task368` | two `QLinearConv` stages plus direct non-gray `Where` bbox mask | `3599 -> 2536` | `+0.350068` |
| `task377` | float16 output tail | `4567 -> 3967` | `+0.140846` |

## Why task368 improved

ARC-GEN task `e76a88a6` places one colored rectangular sprite and several gray
copies. The original graph used a float16 convolution chain to find copy
corners and another float16 convolution to stamp the recovered color template.

The rewrite uses a fixed int8 kernel in `QLinearConv` for corner detection and
a dynamic uint8 template in a second `QLinearConv` for stamping. It also
replaces `Not -> And -> Cast` in bbox detection with
`Where(is_gray, 0, label)`. This preserves the generator rule while removing
more than one thousand units of scorer cost.

Validation for the final task368 graph:

- Official train, test, and bundled ARC-GEN examples: passed.
- Random equivalence checks: 2048, zero candidate mismatches.
- Fresh exact-generator checks: 5000, zero baseline differences and zero
  expected-output failures.
- Online isolated probe ref `54568248`: `7268.13`, confirming `QLinearConv`
  compatibility before the final bundle.

## Online ablations

- `task209 + task328`, ref `54568382`: `7267.96`, complete.
- `task368` v1, ref `54568248`: `7268.13`, complete.
- Final seven-task v5, ref `54568534`: `7268.31`, complete.

Do not reuse the uint8 `TopK` experiments:

- `task018`, ref `54568377`: `ERROR`.
- `task076` with repaired shape metadata, ref `54568450`: `ERROR`.
- `task285` with repaired shape metadata, ref `54568393`: `ERROR`.

Those models pass local ONNX Runtime checks, but the Kaggle evaluator rejects
their uint8 `TopK` paths. They are retained under `experiments/` only as
negative evidence.

## Files

- `submission.zip`: full 400-task Kaggle package.
- `overrides/`: the seven B-task ONNX files to graft onto a newer team base.
- `scripts/`: all self-rewrite, ablation, and deterministic bundle scripts.
- `reports/`: local scoring and equivalence reports.
- `experiments/`: locally valid but online-rejected uint8 `TopK` candidates.

