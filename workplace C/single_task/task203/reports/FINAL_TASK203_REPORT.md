# FINAL task203 report

- Rule model: reverse concentric-ring colors by complementary channel frequencies.
- Structural candidate: explicit broadcast equality matrix and Gather permutation.
- Validation: 267/267 exact.
- Official cost: 355 baseline, 795 candidate, delta +440.
- Artifact: `onnx/task203_frequency_involution.onnx`.
- Replacement accepted: no.
- Next useful direction: derive the same involution without materializing the 10x10 equality matrix.
