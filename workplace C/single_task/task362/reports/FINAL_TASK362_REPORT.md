# FINAL task362 report

- Rule model: marker-count displacement of a cross intersection followed by full-line redraw.
- Structural candidate: cross-color count, coordinate moments, target-coordinate comparisons, and one-hot reconstruction.
- Validation: 267/267 exact.
- Official cost: 521 baseline, 18041 candidate, delta +17520.
- Artifact: `onnx/task362_dynamic_shift.onnx`.
- Replacement accepted: no.
- Next useful direction: retain the verified moment formula but encode row/column/color terms as low-width factors, matching the baseline's factorized output cost.
