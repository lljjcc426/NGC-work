# FINAL TASK096 REPORT

Generated: 2026-07-11

## Attempt

The dedicated probe replaced the two full row/column `ReduceSum` operations
with `ReduceMax`, based on the hypothesis that downstream logic only needed
presence flags.

## Result

- Baseline cost: `7678`
- Probe reported cost: `7678`
- Full validation: failed
- Status: rejected

The hypothesis was false. The reductions also encode counts used by dynamic
shape and index calculations. The modified graph produced `19x19` outputs for
some examples and generated out-of-range `ScatterElements` indices for others.

## Decision

Do not replace the row/column sums independently. A future task096 rewrite must
rederive the complete radius, output-size, and scatter-index logic together.
