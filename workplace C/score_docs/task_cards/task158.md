# task158 Score Card

Generated: 2026-07-09T15:34:05

## Score Priority

- priority_band: `P0_lt16`
- assignment_cost: `28483`
- assignment_points: `14.742937`
- current_cost: `28483.0`
- current_score: `14.742937302942996`
- quick depth: `deep P0/P1`

## Why This Task Matters

- C track role: `onnx_equiv_compression`.
- Score gap focus: cost compression on current ONNX artifact; no rule rewrite is accepted without full validation.

## Structure

- shape_class: `same_shape`
- size_trend: `same_size`
- color_class: `input_palette_only`
- train/test/arc-gen: `3/1/262`
- input_shapes: `14x15;15x15;15x16;16x15;16x16;16x17;17x16;17x17;17x18;18x17;18x18;18x19;19x18;19x19;19x20;20x19;20x20;20x21;21x20;21x21;21x22;22x21;22x22;22x23;23x22;23x23;23x24;24x23;24x24;24x25;25x24;25x25;26x25`
- output_shapes: `14x15;15x15;15x16;16x15;16x16;16x17;17x16;17x17;17x18;18x17;18x18;18x19;19x18;19x19;19x20;20x19;20x20;20x21;21x20;21x21;21x22;22x21;22x22;22x23;23x22;23x23;23x24;24x23;24x24;24x25;25x24;25x25;26x25`
- same_shape_all_examples: `True`
- output_colors_subset_input: `True`
- avg_changed_cell_ratio_same_shape: `0.0418`

## Pattern Understanding

- Same-shape sparse overlay task.
- Verified rule: detect dominant background, find a 3x3 motif with two opposite-corner endpoint colors and one fill color, then overlay rotated/reflected/scaled motif fill masks between compatible square marker-color pairs.
- Python rule solver passes `266/266` examples: train `3/3`, test `1/1`, arc-gen `262/262`.
- Output palette is input-contained; only sparse fill-mask construction needs modification.

## ONNX Compression Opportunities

- Baseline ONNX is task-specific and already implements motif anchor detection plus scale 1/2/3 square marker-pair stamping.
- Accepted compression: replace scale-2 and scale-3 expanded stamp `Gather` index tensors with nearest `Resize` from the scale-1 orientable stamp mask.
- This preserves rule behavior and removes `460` counted params.

## Concrete Next Experiments

1. Build a 400-file candidate package from current best artifacts and replace only `task158.onnx`.
2. Validate `file_count=400`, `missing_task_count=0`, and run local package checks before any Kaggle submission.
3. Try the same `expanded stamp Gather -> Resize` rewrite on other C motif/stamp tasks, starting with `task286` and `task054`.

## Cost Diff

| attempt | old_cost | new_cost | delta_cost | local_valid | accepted | artifact_path |
| --- | ---: | ---: | ---: | --- | --- | --- |
| task158_20260709T205621 | 28483 | 28023 | 460 | True | True | `workplace C\single_task\task158\onnx\task158_candidate.onnx` |

## Attempts

- `task158_resize_stamp_builder`: accepted. Full local validation passed on `266/266` examples.
- Baseline self-check: `28483 -> 28483`, valid, not accepted.

## Next Best Action

- Package the accepted `task158` replacement into a full candidate zip after checking all 400 task files.
