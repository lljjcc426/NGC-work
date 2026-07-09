# task383 Score Card

Generated: 2026-07-09T15:34:05

## Score Priority

- priority_band: `P1_16_16p7`
- assignment_cost: `5912`
- assignment_points: `16.315261`
- current_cost: `5830.0`
- current_score: `16.32922772065546`
- quick depth: `deep P0/P1`

## Why This Task Matters

- C track role: `onnx_equiv_compression`.
- Score gap focus: cost compression on current ONNX artifact; no rule rewrite is accepted without full validation.

## Structure

- shape_class: `same_shape`
- size_trend: `same_size`
- color_class: `input_palette_only`
- train/test/arc-gen: `3/1/262`
- input_shapes: `12x15;13x14;13x15;13x16;13x17;13x18;13x19;13x20;13x21;13x22;14x12;14x13;14x14;14x15;14x16;14x17;14x18;14x20;14x21;14x22;14x23;15x12;15x13;15x14;15x16;15x17;15x18;15x19;15x20;15x21;15x22;15x23;15x24;16x13;16x14;16x15;16x16;16x17;16x18;16x19;16x20;16x21;16x22;16x23;17x14;17x15;17x16;17x17;17x18;17x19;17x20;17x21;17x22;17x23;18x12;18x13;18x15;18x17;18x19;18x20;18x21;18x22;18x23;19x12;19x13;19x14;19x15;19x16;19x17;19x18;19x19;19x20;19x21;19x22;19x23;19x24;20x13;20x14;20x15;20x16;20x17;20x18;20x19;20x20;20x21;20x23;21x13;21x14;21x15;21x16;21x17;21x18;21x19;21x20;21x22;22x12;22x13;22x14;22x15;22x16;22x18;22x19;22x20;22x21;22x22;23x14;23x15;23x16;23x17;23x18;23x22;23x23;24x14;24x15;24x17;24x18;24x20`
- output_shapes: `12x15;13x14;13x15;13x16;13x17;13x18;13x19;13x20;13x21;13x22;14x12;14x13;14x14;14x15;14x16;14x17;14x18;14x20;14x21;14x22;14x23;15x12;15x13;15x14;15x16;15x17;15x18;15x19;15x20;15x21;15x22;15x23;15x24;16x13;16x14;16x15;16x16;16x17;16x18;16x19;16x20;16x21;16x22;16x23;17x14;17x15;17x16;17x17;17x18;17x19;17x20;17x21;17x22;17x23;18x12;18x13;18x15;18x17;18x19;18x20;18x21;18x22;18x23;19x12;19x13;19x14;19x15;19x16;19x17;19x18;19x19;19x20;19x21;19x22;19x23;19x24;20x13;20x14;20x15;20x16;20x17;20x18;20x19;20x20;20x21;20x23;21x13;21x14;21x15;21x16;21x17;21x18;21x19;21x20;21x22;22x12;22x13;22x14;22x15;22x16;22x18;22x19;22x20;22x21;22x22;23x14;23x15;23x16;23x17;23x18;23x22;23x23;24x14;24x15;24x17;24x18;24x20`
- same_shape_all_examples: `True`
- output_colors_subset_input: `True`
- avg_changed_cell_ratio_same_shape: `0.1091`

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

1. `python "workplace C/neurogolf-2026-work/scripts/c_score_scan_artifacts.py" --tasks task383 --score-top-n 8 --full-validate`
2. Inspect best lower-size artifacts in `workplace C/score_docs/artifact_scans/` and compare memory/params split for `task383`.
3. If a lower-cost artifact validates, register it with `c_cost_diff_runner.py --task task383 --old-artifact <current> --new-artifact <candidate> --accept-if-better`.

## Cost Diff

| attempt | old_cost | new_cost | delta_cost | local_valid | accepted | artifact_path |
| --- | ---: | ---: | ---: | --- | --- | --- |
| pending |  |  |  |  |  |  |

## Attempts

- No accepted C-local attempt recorded yet.

## Next Best Action

- Run artifact scan and accept the first full-validation lower-cost artifact.
