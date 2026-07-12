# FINAL task373 report

- Rule: construct a 2x6 checkerboard from the two input row colors.
- Candidate: `onnx/task373_candidate.onnx`.
- Structure: two source Slice nodes and task-specific alternating Concat rows.
- Full validation: 75/75.
- Baseline cost: 60.
- Candidate cost: 1058.
- Accepted replacement: no. The independent graph is exact but loses the parent's zero-intermediate Einsum advantage.
