# Submission 4 B exact batch 2

This round continues from the hidden-safe nine-task package in
`20260715_b_submission4_original_batch1_v1`. It changes only B tasks 063, 185,
and 295. No public model package or non-B task was blended.

## Exact rewrites

| Task | Method | Cost | Local gain |
| --- | --- | ---: | ---: |
| `task063` | store the row/column 0/1/2 class chain as uint8 instead of float16 | `1706 -> 1556` | `+0.092033` |
| `task185` | replace bool-to-int8 casts and ramp multiplication with uint8 `Where` ramps | `1682 -> 1623` | `+0.035707` |
| `task295` | use opset-14 uint8 arithmetic for exact integer triangle geometry | `1604 -> 1587` | `+0.010655` |

All three models pass their complete train, test, and 262-example ARC-GEN
datasets. The changes preserve the existing analytic rules; they do not fit
example outputs.

## Cumulative candidate

- Team submission 4 local baseline: `7386.993113263445`.
- Previous hidden-safe nine-task gain: `+0.882549898892`.
- New three-task gain: `+0.138395364232`.
- Cumulative B-only gain: `+1.020945263124`.
- Projected local score: `7388.014058526569`.
- Submission SHA256:
  `B00C5AFDF426D175E4DC2ECD3ACA7F0046E3A2AB1391809CC572F3B87FE97779`.

The ZIP contains exactly `task001.onnx` through `task400.onnx` at archive
root and passes CRC validation. This crosses the agreed `+1.0` threshold and
was submitted for direct Kaggle verification.

## Kaggle result

- Ref: `54733885`.
- Status: `COMPLETE`.
- Public score: `7388.17`.
- Same-base control: ref `54711326`, public score `7387.15`.
- Confirmed online gain: `+1.02`.

The online delta matches the local `+1.020945` prediction at leaderboard
precision. The three new rewrites are therefore accepted as hidden-safe.

## Contents

- `models/`: the three new B-task overrides.
- `scripts/`: reproducible structural rewrite scripts.
- `reports/summary.json`: exact costs, gains, and package identity.
- `submission/submission.zip`: cumulative 12-task candidate on submission 4.
