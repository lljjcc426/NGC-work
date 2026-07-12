# FINAL task072 report

- Rule model: cellwise XOR of the two pictures around the separator.
- Structural candidate: one 8x1 Conv computes bottom-minus-top directly.
- Validation: 268/268 exact.
- Official cost: 421 baseline, 368 candidate, delta -53.
- Delta points: `+0.13454989551345165`.
- Artifact: `onnx/task072_candidate.onnx`.
- Replacement accepted: yes, locally.
- Next useful direction: reduce the 80-element dense Conv initializer without materializing a full input-channel slice.
