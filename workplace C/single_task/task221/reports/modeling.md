# task221 independent modeling

## Rule

Let `z` be the number of color-0 cells in the 3x3 motif and `f = 9-z` the number of foreground cells. The output canvas is a `z by z` grid of 3x3 blocks. The original motif is copied into the first `f` block positions in row-major order; remaining in-canvas blocks are color 0. Cells outside the `3z by 3z` canvas are benchmark padding.

## Structural candidate

`scripts/build_tile_rank_fill.py` implements the rule directly:

1. Count zero cells in the cropped motif.
2. Compute each output cell's block rank as `block_row*z + block_col`.
3. Mark ranks below `f` as active and rows/columns below `z` as in-canvas.
4. `Tile` the motif 10x10, select it for active blocks, select channel 0 for inactive in-canvas blocks, and zero benchmark padding.

This is a direct rank-and-tile model, structurally unrelated to the baseline's factorized eight-input Einsum.

## Result

- Baseline: 267/267, cost 805.
- Candidate: 267/267, cost 78973.
- Decision: reject for replacement. The rule is exact, but the tiled `10x30x30` motif and full-grid masks are extremely expensive. The result validates why the baseline factorizes row, column, and motif terms instead of materializing the canvas.
