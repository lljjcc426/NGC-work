# FINAL task113 report

- Rule: reflect the first five rows to fill a fixed ten-row canvas.
- Candidate: `onnx/task113_candidate.onnx`.
- Structure: Slice + reverse Slice + Concat + Pad.
- Full validation: 265/265.
- Baseline cost: 30.
- Candidate cost: 24014.
- Accepted replacement: no. The direct geometric model is exact, but its intermediate activations are much more expensive than the output-direct Gather.
