# FINAL TASK077 REPORT

Generated: 2026-07-12

## Accepted result

- Current effective artifact: cost 7655, 266/266.
- Candidate: `workplace C/single_task/task077/onnx/task077_candidate.onnx`
- Candidate validation: 266/266.
- Candidate memory: 7200.
- Candidate parameters: 34.
- Candidate cost: 7234.
- Cost delta: -421.
- Candidate points: 16.113452587487956.
- Point delta: +0.056566895579742.
- Accepted: true.

## Implemented model

The candidate keeps the required three rounds of barrier-constrained horizontal
propagation. It replaces the final quantized vertical kernel `[1,3,1]` with
`[1,2,1]`, making every positive bridge response exactly one. The exclusion of
source color-2 pixels can then be expressed as `S > R`, deleting the full
`T=2R` tensor and its quantized 1x1 convolution.

This is a task-specific quantized logic fusion, not a generic optimizer, opset,
Pad-only, or initializer-only change.

## Rejected alternatives

- Ordinary two-round propagation: best 258/266.
- 55 two-round kernel/dilation schedules: best 227/266.
- Directly removing vertical bridge `S`: one example fails by five cells.
- Large single-step morphology: crosses disconnected barrier components.

No parent package, Kaggle kernel, or submission was built.
