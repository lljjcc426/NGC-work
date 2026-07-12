# FINAL task121 report

- Rule model: marker-centered object extraction and body-color repaint.
- Structural candidate: coordinate locator, dynamic 3x3 Slice, occupancy reduction, and Where reconstruction.
- Validation: 266/266 exact.
- Official cost: 326 baseline, 4582 candidate, delta +4256.
- Artifact: `onnx/task121_marker_object.onnx`.
- Replacement accepted: no.
- Next useful direction: fuse marker localization and body-color selection into one low-width checksum while preserving the verified object rule.
