# task144 independent modeling

## Rule

The input contains a top 4x4 color-7 pattern, a color-4 separator, and a bottom 4x4 color-2 pattern. The output marks color 3 exactly where both aligned pattern cells are blank. All other cells become color 0. In one-hot terms, the color-3 plane is the intersection of the two channel-0 background planes.

## Alternative computation

`task144_background_intersection.onnx` slices channel 0 from the top and bottom patterns, multiplies the two background masks, uses the complement for output channel 0, explicitly constructs output channel 3, and pads the result. This is a direct set-intersection model, not a decomposition of the parent dilated Conv weights.

## Validation and cost

- Examples: 4 train, 1 test, 262 arc-gen.
- Baseline grouped Conv: 267/267, memory 0, params 104, cost 104.
- Background-intersection candidate: 267/267, memory 960, params 23, cost 983.
- Decision: mathematically equivalent alternative completed; replacement rejected, delta cost +879.
