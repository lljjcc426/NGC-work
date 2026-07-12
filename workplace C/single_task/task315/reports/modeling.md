# task315 independent modeling

## Rule

The input is a 3x3 pattern using colors 0, 1, and 2. The output is a 3x3 grid of 3x3 tiles. At every input position containing color 2, the complete original input pattern is copied into the corresponding output tile; positions not containing color 2 receive an all-background tile. Algebraically, this is the Kronecker product of the color-2 position mask with the original one-hot pattern, with background reconstruction for inactive tiles.

## Alternative computation

`task315_kronecker.onnx` extracts the three relevant channels and the color-2 mask, reshapes them into broadcast-compatible six-dimensional tensors, selects between the original pattern and a background tile with Where, transposes block and within-block axes, and reshapes to 9x9. This replaces the ten-factor direct-output Einsum with an explicit Kronecker construction.

## Validation and cost

- Examples: 3 train, 1 test, 262 arc-gen.
- Baseline Einsum: 266/266, memory 0, params 230, cost 230.
- Kronecker candidate: 266/266, memory 3213, params 39, cost 3252.
- Decision: mathematically equivalent alternative completed; replacement rejected, delta cost +3022.
