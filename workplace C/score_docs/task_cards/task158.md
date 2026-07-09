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

- Same-shape task. Prioritize mask/color logic compression and removal of redundant full-grid branches.
- Output palette is input-contained; unused color creation branches are likely removable.

## ONNX Compression Opportunities

- Run artifact scan against all local public and candidate ONNX sources.
- Fully validate any lower-cost artifact on train + test + arc-gen before accepting.
- Compare official memory/params split to locate whether memory graph or constants dominate.
- Try same-shape fast path: simplify masks, color comparisons, and identity-preserving branches.
- Exploit input-palette-only constraint; remove unused output color branches and redundant compares.
- Changed-cell ratio is low; test sparse mask overlay instead of full-grid recompute.

## Concrete Next Experiments

1. `python "workplace C/neurogolf-2026-work/scripts/c_score_scan_artifacts.py" --tasks task158 --score-top-n 8 --full-validate`
2. Inspect best lower-size artifacts in `workplace C/score_docs/artifact_scans/` and compare memory/params split for `task158`.
3. If a lower-cost artifact validates, register it with `c_cost_diff_runner.py --task task158 --old-artifact <current> --new-artifact <candidate> --accept-if-better`.

## Cost Diff

| attempt | old_cost | new_cost | delta_cost | local_valid | accepted | artifact_path |
| --- | ---: | ---: | ---: | --- | --- | --- |
| pending |  |  |  |  |  |  |

## Attempts

- No accepted C-local attempt recorded yet.

## Next Best Action

- Run artifact scan and accept the first full-validation lower-cost artifact.
