# FINAL task146 report

- Rule model: select the unique non-transpose-symmetric 3x3 tile.
- Structural candidate: explicit triangle-pair comparison and dynamic tile Gather.
- Validation: 267/267 exact.
- Official cost: 265 baseline, 7765 candidate, delta +7500.
- Artifact: `onnx/task146_asymmetry.onnx`.
- Replacement accepted: no.
- Next useful direction: retain the explicit symmetry rule but compress the three pair checks into a low-width checksum without full Gather activations.
