# FINAL task351 report

- Rule modeled: color-3 row/column locator followed by reverse 5x5 extraction.
- Independent ONNX: `onnx/task351_candidate.onnx`.
- Validation: 265/265 across train, test, and arc-gen.
- Baseline cost: 1229.
- Candidate cost: 3863.
- Accepted: no; exact but memory-expensive.
- Best next structure: compute channel-3 row and column reductions directly without materializing all ten color reductions.
