# task180 independent modeling

## Rule

The 8x8 input consists of four aligned 4x4 binary layers: color 4 in the upper-left, color 5 in the upper-right, color 6 in the lower-left, and color 9 in the lower-right. At each output cell the layers are overlaid with strict priority `5 > 6 > 9 > 4 > 0`.

The priority table was checked over all public examples. For example, color 5 wins every overlap; without 5, color 6 wins; without 5 or 6, color 9 wins over 4.

## Alternative computation

`task180_priority_overlay.onnx` slices the four color planes independently, converts them to boolean masks, computes the priority exclusions with Not/And, concatenates the mutually exclusive output channels, and pads the 4x4 result. It replaces the grouped dilated Conv with an explicit logical decision network.

## Validation and cost

- Examples: 5 train, 1 test, 262 arc-gen.
- Baseline grouped Conv: 268/268, memory 0, params 210, cost 210.
- Priority overlay candidate: 268/268, memory 1296, params 34, cost 1330.
- Decision: mathematically equivalent alternative completed; replacement rejected, delta cost +1120.
