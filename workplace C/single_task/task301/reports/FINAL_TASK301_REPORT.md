# FINAL task301 report

- Rule modeled: length-ranked, right-aligned colored bars.
- Independent ONNX: `onnx/task301_candidate.onnx`.
- Validation: 266/266 across train, test, and arc-gen.
- Baseline cost: 1141.
- Candidate cost: 36751.
- Accepted: no; exact but full-grid masks are too expensive.
- Best next structure: retain a single final comparison tensor and encode row/color thresholds without separate Boolean masks.
