# FINAL task178 report

- Rule modeled: orientation selection plus run-length compression of the first active line.
- Independent ONNX: `onnx/task178_candidate.onnx`.
- Validation: 268/268 across train, test, and arc-gen.
- Baseline cost: 762.
- Candidate cost: 73884.
- Accepted: no; exact but full-grid ArgMax is too expensive.
- Best next structure: keep first-line-only label extraction and reduce the downstream run-selection graph without expanding spatially.
