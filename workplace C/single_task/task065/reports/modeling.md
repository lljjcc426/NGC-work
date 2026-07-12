# task065 independent modeling

## Rule

The odd square input is divided into four equal quadrants by a full middle row and column. All quadrants share a dominant background and exactly one quadrant contains a single rare-color dot. Removing the divider and folding the four quadrants onto one another produces an `m x m` output, where `m=(side-1)/2`; the dot coordinate is reduced modulo `m+1`.

## Candidate structure

`task065_fold_scatter.onnx` computes the dynamic side and output size, masks a fixed 7x7 top-left canvas to the valid `m x m` region, identifies the count-one color, finds its row and column with coordinate moments, folds the coordinates modulo `m+1`, and overwrites the corresponding NHWC cell with ScatterND. This is a direct fold-and-overlay model rather than the baseline's bit-field synthesis.

The candidate uses a fixed 7x7 internal canvas so every intermediate has a static shape and official memory scoring remains defined.

## Validation

- Public examples: 3 train, 1 test, 262 arc-gen.
- Candidate: 266/266 exact.
- Baseline cost: 638 (memory 582, params 56).
- Candidate cost: 14196 (memory 14129, params 67).
- Decision: rule accepted as an independent model; replacement rejected because explicit canvas and ScatterND tensors are expensive.
