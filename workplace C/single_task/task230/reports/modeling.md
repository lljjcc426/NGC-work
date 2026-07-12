# task230 independent modeling

## Rule

Every object is a solid 2x2 block of color 5. Four marker cells are placed one cell diagonally outside the block: color 1 at the upper-left, color 2 at the upper-right, color 3 at the lower-left, and color 4 at the lower-right. The original block remains color 5. Multiple blocks are processed independently.

## Alternative computation

`task230_pool_deconv.onnx` is not a sparse rewrite of the parent Conv. It performs 2x2 AveragePool on the color-5 plane, tests for an exact all-5 block, and feeds the block-origin mask to a four-output ConvTranspose whose four 4x4 kernels contain one corner impulse each. A crop aligns those impulses with offsets `(-1,-1)`, `(-1,+2)`, `(+2,-1)`, and `(+2,+2)`. The original color-5 and background planes are then combined with the four generated marker planes.

The first implementation incorrectly generated background outside the true grid. The final model reconstructs background from the original channel-0 plane and subtracts generated marker occupancy.

## Validation and cost

- Examples: 3 train, 1 test, 262 arc-gen.
- Baseline Conv: 266/266, memory 0, params 900, cost 900.
- Pool plus ConvTranspose candidate: 266/266, memory 70753, params 75, cost 70828.
- Decision: mathematically equivalent alternative completed; replacement rejected, delta cost +69928.
