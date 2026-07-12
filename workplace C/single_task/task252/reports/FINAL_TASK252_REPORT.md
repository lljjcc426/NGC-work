# FINAL task252 report

- Rule: recolor nonzero cells in odd columns to color 4; preserve all other cells.
- Candidate: `onnx/task252_candidate.onnx`.
- Structure: background/foreground channel split, explicit parity masks, foreground occupancy reduction, color-4 branch merge.
- Full validation: 266/266.
- Baseline cost: 180.
- Candidate cost: 219689.
- Accepted replacement: no. The model is exact but full-grid branch activations are far more expensive than the fused Einsum.
