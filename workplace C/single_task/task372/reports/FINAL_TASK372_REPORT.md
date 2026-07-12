# FINAL TASK372 REPORT

- Candidate: public-exact `7x1 group=2 Conv`.
- Validation: `266/266`.
- Parent/candidate cost: `710 -> 360`.
- Parent/candidate points: `18.43473502996464 -> 19.113895968549844`.
- Delta points: `+0.6791609385852044`.
- Official memory: `0`.
- Artifact: `onnx/task372_candidate.onnx`.
- Decision: local accepted.

The earlier crop/group/pad model was exact but memory-negative. The accepted
builder instead solves hard-margin constraints over every public vertical
window and preserves the one-node graph. `group=2` is exact; `group=5` and
`group=10` are infeasible. Sampling only rows 0 and 6, or dilation patterns
`[0,3,6]` and `[0,2,4,6]`, fails colors 6..9, so the complete seven-row support
is retained.
