# FINAL TASK180 REPORT

- Rule understood: aligned four-layer overlay with priority `5 > 6 > 9 > 4 > 0`.
- Independent structure: four Slice operations plus an explicit boolean priority network.
- Full validation: 268/268 exact.
- Baseline cost: 210.
- Candidate cost: 1330.
- Delta cost: +1120.
- Candidate: `onnx/task180_priority_overlay.onnx`.
- Accepted replacement: no.
- Result: the logical factorization is exact and uses fewer parameters, but its intermediate masks cost more than the direct grouped Conv.
