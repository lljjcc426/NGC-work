# task052 independent modeling

## Rule

The 3x3 input is processed row by row. A row is painted entirely color 5 when all three cells in that row have the same input color; every other row is painted color 0. The output never preserves the original palette.

## Candidate structure

`task052_monochrome_rows.onnx` crops the 3x3 active region, applies a grouped 1x3 convolution independently to all ten one-hot channels, detects a channel count of three, reduces that evidence to one boolean per row, and broadcasts the resulting color-0/color-5 selector across three columns. This is a direct row-classification model and does not reuse the baseline Einsum/hash construction.

## Validation

- Public examples: 4 train, 1 test, 262 arc-gen.
- Candidate: 267/267 exact.
- Baseline cost: 194 (memory 90, params 104).
- Candidate cost: 811 (memory 768, params 43).
- Decision: rule accepted as an independent model; ONNX replacement rejected because cost increased by 617.
