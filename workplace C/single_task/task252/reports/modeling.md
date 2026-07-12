# task252 independent modeling

## Rule

Every nonzero cell in an even zero-based column retains its input color. Every nonzero cell in an odd column becomes color 4. Color-0 background cells remain color 0. The row index and neighboring cells do not affect the transformation.

## Independent structure

`scripts/build_odd_column_recolor.py` replaces the parent's channel/parity Einsum with explicit semantic branches:

1. Slice and preserve channel 0 for all columns.
2. Slice channels 1 through 9 and retain them only under an even-column mask.
3. Reduce the foreground channels to occupancy, select odd columns, and place that occupancy in channel 4.
4. Add the three disjoint one-hot branches.

This is a task-specific channel-routing graph and does not reuse the parent's `P`, `Q`, or `par` factors.

## Official validation and cost

| variant | passed | checked | memory | params | cost |
|---|---:|---:|---:|---:|---:|
| channel/parity Einsum | 266 | 266 | 0 | 180 | 180 |
| explicit parity branches | 266 | 266 | 219600 | 89 | 219689 |

The branch model is exact and halves parameter storage, but full-grid foreground tensors dominate official memory cost. It is a valid independent model, not an accepted replacement.
