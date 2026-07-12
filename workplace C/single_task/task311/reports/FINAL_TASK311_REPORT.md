# FINAL task311 report

- Rule: append a horizontal reflection of every input row.
- Candidate: `onnx/task311_candidate.onnx`.
- Structure: crop Slice + reverse Slice + Concat + Pad.
- Full validation: 266/266.
- Baseline cost: 30.
- Candidate cost: 1458.
- Accepted replacement: no. The candidate is exact but cannot beat a direct-output Gather with zero activation cost.
