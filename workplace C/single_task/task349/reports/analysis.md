# task349 Analysis

- Public examples: 4 train, 1 test, 262 arc-gen, 267 total.
- Input/output shapes are equal and range from 10x10 to 30x30.
- Inputs contain one or more solid color-9 rectangles with even widths 2, 4, 6, 8, or 10.
- Each rectangle receives a color-3 halo whose radius is half its width.
- Color-1 rays extend downward from color-9 cells through background.
- Precedence is color 9 over color 3 over color 1 over background.
- The Python rule validates 267/267 examples.

The baseline ONNX detects each rectangle width in a separate channel, producing
a 5x30x30 byte tensor, then applies five nested halo kernels. Packed low-rank
encodings were evaluated but rejected after full overlap testing. The accepted
candidate instead preserves exact width classes while shrinking their spatial
activation to 5x29x29.

## 2026-07-12 accepted spatial width-map compression

The aggressive packed-width candidates were not full-validation safe on arc-gen
overlap cases. The accepted local candidate now uses a baseline-equivalent
compression instead: the horizontal detector kernel is shortened from 12 to 11
columns by removing the redundant right-boundary check after width 10, the
maximum task width.

- Original baseline: 267/267, cost 14892.
- Prior width-11 candidate: 267/267, cost 14887.
- Final `5x29x29` width-map candidate: 267/267, cost 14647.
- Delta versus original: -245 cost, +0.01658861967429992 points.

The final candidate removes the impossible last width-map column and the
redundant top trigger row, then compensates in halo padding. Bottom-row removal
was tested and rejected because bottom-clipped height-1 rectangles occur in the
public generator set.
