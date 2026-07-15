# E task064 complement-equal online result

This folder adds a new `task064` graph rewrite to the exact E package scored at
`7385.77`.

## Rewrite

The source model uses uint8 row and column state codes. The accepted rewrite
keeps those codes and the original `hline` and `vline` expressions. Because the
two line tensors are mutually exclusive, equality means no line is required.
The rewrite replaces terminal `Max + Cast` with `Equal` and reverses the final
`Where` branches.

- Cost: `8356 -> 7507`.
- Local gain: `+0.107144`.
- Official validation: `267/267` examples passed.
- Differential color-permutation validation: `0/5000` mismatches.
- Model SHA256:
  `473256223d033d3d7564c43fcb2e52bc15e119cbd2e1ea151f7b741457250146`.

## Kaggle result

- Parent ref and score: `54708238`, `7385.77`.
- Submission ref: `54708724`.
- Status: `COMPLETE`.
- Public score: `7385.88`.
- Displayed online gain: `+0.11`.
- Submission ZIP SHA256:
  `c2196b88f5293b545dd089c802b327408ccb6e37f7841846aec6e00873e1032b`.

## Rejected experiment

An initial reconstruction treated the uint8 row and column state tensors as
ordinary booleans. It failed all `267` official examples and was overwritten
locally before any GitHub commit or Kaggle submission. The accepted script in
`../e_optimize_task064_complement_equal_20260715.py` preserves the encoded
state semantics.

The cumulative gain over team `7384.93` is now `+0.95`.
