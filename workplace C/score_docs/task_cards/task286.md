# task286 Score Card

Generated: 2026-07-09T15:34:05

## Score Priority

- priority_band: `P0_lt16`
- assignment_cost: `26909`
- assignment_points: `14.799784`
- current_cost: `26909.0`
- current_score: `14.799783917876258`
- quick depth: `deep P0/P1`

## Why This Task Matters

- C track role: `onnx_equiv_compression`.
- Score gap focus: cost compression on current ONNX artifact; no rule rewrite is accepted without full validation.

## Structure

- shape_class: `same_shape`
- size_trend: `same_size`
- color_class: `input_palette_only`
- train/test/arc-gen: `2/1/262`
- input_shapes: `10x11;10x12;10x15;10x16;10x18;10x19;10x22;10x25;11x14;11x15;11x16;11x17;11x18;11x20;11x21;11x22;11x23;11x24;11x25;12x10;12x11;12x12;12x13;12x15;12x16;12x18;12x19;12x21;12x22;12x23;13x10;13x11;13x12;13x13;13x14;13x15;13x16;13x17;13x18;13x19;13x20;13x21;13x22;13x24;13x25;14x10;14x12;14x17;14x19;14x20;14x22;14x23;14x24;15x10;15x12;15x14;15x15;15x18;15x19;15x20;15x21;15x22;16x10;16x11;16x12;16x13;16x15;16x16;16x17;16x18;16x19;16x20;16x21;16x22;16x23;16x24;16x25;17x10;17x12;17x13;17x14;17x18;17x20;17x21;17x22;17x24;17x25;18x10;18x11;18x12;18x13;18x15;18x16;18x19;18x21;18x23;18x24;19x11;19x13;19x16;19x17;19x18;19x21;19x23;19x24;20x11;20x12;20x13;20x14;20x15;20x18;20x19;20x21;20x22;20x24;21x10;21x11;21x12;21x13;21x15;21x17;21x18;21x19;21x21;21x22;21x23;22x11;22x12;22x17;22x18;22x19;22x20;22x21;22x23;22x24;23x11;23x12;23x13;23x14;23x15;23x16;23x17;23x18;23x19;23x23;23x24;23x25;24x10;24x11;24x12;24x13;24x14;24x18;24x19;24x20;24x22;24x25;25x12;25x13;25x14;25x16;25x17;25x18;25x19;25x21;25x24;25x25`
- output_shapes: `10x11;10x12;10x15;10x16;10x18;10x19;10x22;10x25;11x14;11x15;11x16;11x17;11x18;11x20;11x21;11x22;11x23;11x24;11x25;12x10;12x11;12x12;12x13;12x15;12x16;12x18;12x19;12x21;12x22;12x23;13x10;13x11;13x12;13x13;13x14;13x15;13x16;13x17;13x18;13x19;13x20;13x21;13x22;13x24;13x25;14x10;14x12;14x17;14x19;14x20;14x22;14x23;14x24;15x10;15x12;15x14;15x15;15x18;15x19;15x20;15x21;15x22;16x10;16x11;16x12;16x13;16x15;16x16;16x17;16x18;16x19;16x20;16x21;16x22;16x23;16x24;16x25;17x10;17x12;17x13;17x14;17x18;17x20;17x21;17x22;17x24;17x25;18x10;18x11;18x12;18x13;18x15;18x16;18x19;18x21;18x23;18x24;19x11;19x13;19x16;19x17;19x18;19x21;19x23;19x24;20x11;20x12;20x13;20x14;20x15;20x18;20x19;20x21;20x22;20x24;21x10;21x11;21x12;21x13;21x15;21x17;21x18;21x19;21x21;21x22;21x23;22x11;22x12;22x17;22x18;22x19;22x20;22x21;22x23;22x24;23x11;23x12;23x13;23x14;23x15;23x16;23x17;23x18;23x19;23x23;23x24;23x25;24x10;24x11;24x12;24x13;24x14;24x18;24x19;24x20;24x22;24x25;25x12;25x13;25x14;25x16;25x17;25x18;25x19;25x21;25x24;25x25`
- same_shape_all_examples: `True`
- output_colors_subset_input: `True`
- avg_changed_cell_ratio_same_shape: `0.3532`

## Pattern Understanding

- Same-shape maze/corridor propagation task.
- Verified rule: `8` is wall, `0` is unfilled corridor, the two non `{0,8}` colors are marker colors.
- Output fills only the passable connected component containing markers, alternating marker colors by 4-neighbor graph distance.
- Python rule solver passes `265/265` examples: train `2/2`, test `1/1`, arc-gen `262/262`.

## ONNX Compression Opportunities

- Baseline ONNX is already a task-specific bitset flood-fill graph.
- Current official split: `memory=26064`, `params=845`, `cost=26909`.
- Constant-only sparse initializer surgery was attempted and rejected by ONNX checker/type inference for `Conv`, `Where`, `Pad`, and `MatMulInteger`.
- Next real opportunity is a lower-memory rewrite of the flood-fill bitset graph, not artifact scan or constant-only edits.

## Concrete Next Experiments

1. Build a custom row-bitset flood-fill ONNX from `workplace C/single_task/task286/scripts/solve_task286_rule.py`.
2. Preserve the verified rule but reduce scalar bitwise intermediates below the current `2393` node graph.
3. Use final checkerboard coloring from marker colors after reachability mask construction; avoid per-color propagation branches.

## Cost Diff

| attempt | old_cost | new_cost | delta_cost | local_valid | accepted | artifact_path |
| --- | ---: | ---: | ---: | --- | --- | --- |
| task286_20260709T224049 | 26909 | 26909 | 0 | True | False | `E:\kagglegolf\submissions\candidates\GOLF_20260709_101_prvsiyan_7266_72_repro\onnx\task286.onnx` |

## Attempts

- Python rule solver: accepted as rule model, full local example validation `265/265`.
- Sparse initializer ONNX probe: no accepted candidate; checker/type inference rejects sparse constants for relevant ops.
- Baseline self-check: `26909 -> 26909`, valid, not accepted.

## Next Best Action

- Write a new lower-memory bitset flood-fill ONNX builder; do not repeat artifact scan, onnxsim, or sparse-constant surgery.
