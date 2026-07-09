# FINAL TASK286 REPORT

## Result

- task: `task286`
- Python rule analysis complete: `yes`
- Python rule solver: `workplace C/single_task/task286/scripts/solve_task286_rule.py`
- train pass: `2/2`
- test pass: `1/1`
- arc-gen pass: `262/262`
- total pass: `265/265`
- ONNX candidate generated: `no accepted candidate`
- accepted: `false`

## Cost

| old_cost | new_cost | delta_cost | old_points | new_points | delta_points |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 26909 | 26909 | 0 | 14.799783917876258 | 14.799783917876258 | 0.0 |

## Rule

The task is a same-shape maze/corridor propagation problem:

- `8` is wall/barrier.
- `0` is unfilled corridor.
- The two non `{0,8}` colors are marker colors.
- Only the passable connected component containing marker cells is filled.
- Fill color alternates between the two marker colors by 4-neighbor graph distance from the seed markers.
- Passable components without markers remain `0`.

The input-only BFS solver passes all known examples.

## ONNX Probe

The baseline graph is already a task-specific bitset flood-fill implementation:

- nodes: `2393`
- main ops: `BitwiseAnd`, `BitwiseOr`, `BitShift`
- official score split: `memory=26064`, `params=845`

Sparse initializer conversion was tested for the main zero-heavy constants, but ONNX checker/type inference rejects sparse inputs for `Conv`, `Where`, `Pad`, and `MatMulInteger` in this graph. No sparse candidate reached official cost validation.

## Next

1. Build a new bitset flood-fill graph from the verified Python rule, not by modifying constants.
2. Target fewer scalar bitwise intermediates than the current graph while preserving max observed distance `77`.
3. Try row-bitset BFS with parity-independent reachability mask plus final checkerboard coloring.
4. If a candidate validates, run `c_cost_diff_runner.py` and update the C candidate register.
