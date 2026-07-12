# FINAL TASK096 REPORT

Generated: 2026-07-12

## Accepted Model

The baseline's two `ReduceSum` nodes materialize `10x30` float row and column
projections even though all task096 grids fit inside the first 19 rows and
columns. The accepted model replaces them with two depthwise projection Conv
nodes. Negative trailing pads crop the inactive tail while the kernels sum the
orthogonal axis, producing compact `10x19` projections directly.

The downstream count, radius, output-size, and scatter-index logic is preserved
exactly. This avoids the earlier invalid `ReduceMax` simplification.

## Result

- Baseline: memory 7260, params 418, cost `7678`.
- Candidate: memory 6050, params 800, cost `6850`.
- Cost reduction: `828`.
- Points: `16.0538856244 -> 16.1679960687`.
- Delta points: `+0.1141104443`.
- Full validation: `266/266`.
- Status: local accepted.
- Artifact: `workplace C/single_task/task096/onnx/task096_candidate.onnx`.
