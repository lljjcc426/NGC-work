# task096 independent modeling

The parent uses row and column sums both as presence flags and as counts that
control dynamic output size and scatter indices. The independent candidate
replaces both reductions with `ReduceMax` presence detection.

Full validation fails with incorrect 19x19 shapes and out-of-range scatter
indices. This establishes that any lower-cost rewrite must jointly reconstruct
radius, size, and index logic rather than optimize the reductions locally.
