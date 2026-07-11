# task349 Analysis

- Public examples: 4 train, 1 test, 262 arc-gen, 267 total.
- Input/output shapes are equal and range from 10x10 to 30x30.
- Inputs contain one or more solid color-9 rectangles with even widths 2, 4, 6, 8, or 10.
- Each rectangle receives a color-3 halo whose radius is half its width.
- Color-1 rays extend downward from color-9 cells through background.
- Precedence is color 9 over color 3 over color 1 over background.
- The Python rule validates 267/267 examples.

The baseline ONNX detects each rectangle width in a separate channel, producing a 5x30x30 byte tensor, then applies five nested halo kernels. The candidate packs width class 1..5 into one 30x30 byte tensor. A single nested-ring convolution decodes the class, reducing activation memory and initializer count while preserving the same output graph.

## 2026-07-11 accepted local compression

The aggressive packed-width candidates were not full-validation safe on arc-gen
overlap cases. The accepted local candidate now uses a baseline-equivalent
compression instead: the horizontal detector kernel is shortened from 12 to 11
columns by removing the redundant right-boundary check after width 10, the
maximum task width.

- Baseline: 267/267, cost 14892.
- Candidate: 267/267, cost 14887.
- Delta: -5 cost, +0.00033580711555067077 points.
