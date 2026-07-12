# FINAL task221 report

- Rule modeled: row-major motif copies controlled by zero and nonzero cell counts.
- Independent ONNX: `onnx/task221_candidate.onnx`.
- Validation: 267/267 across train, test, and arc-gen.
- Baseline cost: 805.
- Candidate cost: 78973.
- Accepted: no; exact but explicit Tile is prohibitive.
- Best next structure: improve the factorized representation itself; any full-canvas motif tensor loses decisively.
