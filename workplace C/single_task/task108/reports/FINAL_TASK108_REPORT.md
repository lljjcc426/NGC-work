# FINAL TASK108 REPORT

- Rule understood: sample the odd 5x5 lattice and enlarge each cell to 4x4.
- Independent structure: `Slice(step=2) -> Resize(nearest, x4) -> Pad`.
- Full validation: 266/266 exact.
- Baseline cost: 300.
- Candidate cost: 17018.
- Delta cost: +16718.
- Candidate: `onnx/task108_slice_resize.onnx`.
- Accepted replacement: no.
- Result: explicit resampling is exact, but the 5x5 and 20x20 intermediates cost more than the fused Einsum.
