# FINAL task072 report

- Rule model: cellwise XOR of the two pictures around the separator.
- Structural candidate: single-channel comparison and direct color-0/color-3 reconstruction.
- Validation: 268/268 exact.
- Official cost: 421 baseline, 474 candidate, delta +53.
- Artifact: `onnx/task072_xor.onnx`.
- Replacement accepted: no.
- Next useful direction: combine extraction and XOR in one convolutional/checksum operator to remove the two float Slice activations.
