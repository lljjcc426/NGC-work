# task332 Analysis

- Public examples: 4 train, 1 test, 262 arc-gen, 267 total.
- Python rule: 267/267. Each column has exactly one color-5 cell; recolor it to 3 when `(column + grid_width) % 2 == 1`.
- The baseline already computes width parity on a compact 3x20 representation before one QLinearConv materializes the output.

## 2026-07-11 Direct Parity Probe

The direct one-hot builder generated the same output on every public example, but its Gather/Where/Concat intermediates made memory dominate the score:

| artifact | valid | memory | params | cost | delta points |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | true | 389 | 172 | 561 | 0 |
| direct parity output | true | 6,089 | 61 | 6,150 | -2.394486 |

Conclusion: the dynamic parity rule is solved, but direct output construction is unsuitable for the scorer. Future work must simplify the existing compact QLinearConv graph rather than replace it with elementwise one-hot operations.

## 2026-07-12 compact row-code Conv

The first baseline Einsum only uses rows 0..2 and is immediately sliced from
30 to 20 columns. A 3x1 Conv with pads `[0,0,-27,-10]` performs both operations
in one node. It passes 267/267 and lowers official cost from 561 to 438.

A complete hard-margin search over final vertical kernels of heights 1..4 found
no separable replacement; the retained 5x1 QLinearConv has the minimum tested
support for public correctness.
