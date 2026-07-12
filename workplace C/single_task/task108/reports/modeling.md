# task108 independent modeling

## Rule

The meaningful input cells are the 5x5 lattice at rows and columns `1,3,5,7,9` of the 10x10 grid. The output extracts that lattice and expands every sampled cell to a constant 4x4 block, yielding a 20x20 nearest-neighbor image.

## Alternative computation

`task108_slice_resize.onnx` uses a strided Slice with step 2 to obtain the odd-coordinate 5x5 lattice, applies nearest-neighbor Resize with scale 4 in both spatial dimensions, then pads the canonical output tensor to 30x30. This replaces the single five-factor Einsum with explicit sampling and resampling operators.

## Validation and cost

- Examples: 3 train, 1 test, 262 arc-gen.
- Baseline Einsum: 266/266, memory 0, params 300, cost 300.
- Slice plus Resize candidate: 266/266, memory 17000, params 18, cost 17018.
- Decision: mathematically equivalent alternative completed; replacement rejected, delta cost +16718.
