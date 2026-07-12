# FINAL task081 report

- Rule model: complete each cyan L triomino's missing 2x2 corner with color 1.
- Structural candidate: two-layer clipped float convolutional rule network.
- Validation: 264/264 exact.
- Official cost: 464 baseline, 6452 candidate, delta +5988.
- Artifact: `onnx/task081_float_conv.onnx`.
- Replacement accepted: no.
- Next useful direction: preserve quantized arithmetic and search for a one-stage threshold encoding of the two missing-corner orientations.
