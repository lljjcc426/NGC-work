# FINAL task052 report

- Rule model: monochromatic-row detector.
- Structural candidate: grouped horizontal convolution plus row decision and color reconstruction.
- Validation: 267/267 exact.
- Official cost: 194 baseline, 811 candidate, delta +617.
- Artifact: `onnx/task052_monochrome_rows.onnx`.
- Replacement accepted: no.
- Next useful direction: encode the row equality predicate without materializing the cropped 10-channel area; the baseline remains substantially better.
