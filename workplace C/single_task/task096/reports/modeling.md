# task096 independent modeling

The parent uses row and column sums both as presence flags and as counts that
control dynamic output size and scatter indices. The independent candidate
replaces both reductions with `ReduceMax` presence detection.

Full validation fails with incorrect 19x19 shapes and out-of-range scatter
indices. This establishes that any lower-cost rewrite must jointly reconstruct
radius, size, and index logic rather than optimize the reductions locally.

## 2026-07-12 compact projection Conv

The successful rewrite keeps exact counts but changes where they are computed.
Two `group=10` Conv nodes use 19-cell all-one kernels and negative tail pads to
produce `1x10x19x1` and `1x10x1x19` projections. Small Squeeze nodes restore the
existing `5x19` downstream representation after top-color selection.

This preserves all dynamic radius and scatter calculations, passes 266/266,
and lowers official cost from 7678 to 6850.
