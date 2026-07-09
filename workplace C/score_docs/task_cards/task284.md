# task284 Score Card

Generated: 2026-07-09T15:34:05

## Score Priority

- priority_band: `P2_16p7_17p5`
- assignment_cost: `3608`
- assignment_points: `16.809091`
- current_cost: `3089.0`
- current_score: `16.96439730708142`
- quick depth: `light P2/P3`

## Why This Task Matters

- C track role: `onnx_equiv_compression`.
- Score gap focus: cost compression on current ONNX artifact; no rule rewrite is accepted without full validation.

## Structure

- shape_class: `same_shape`
- size_trend: `same_size`
- color_class: `input_palette_only`
- train/test/arc-gen: `3/1/262`
- input_shapes: `10x10;10x11;10x12;10x13;10x14;10x15;10x16;10x17;10x18;10x19;10x20;10x21;10x22;10x24;10x7;10x8;11x10;11x7;11x8;11x9;12x10;12x7;12x8;12x9;13x10;13x7;13x8;14x10;14x7;14x8;15x10;15x7;15x9;16x10;16x7;16x8;16x9;17x10;17x7;17x8;17x9;18x10;18x8;18x9;19x10;19x7;19x8;19x9;20x10;20x7;20x8;20x9;21x7;21x8;21x9;22x10;22x7;22x8;22x9;23x10;23x8;7x10;7x11;7x12;7x13;7x14;7x15;7x16;7x17;7x18;7x19;7x20;7x21;7x22;7x24;7x9;8x10;8x11;8x12;8x13;8x14;8x15;8x16;8x18;8x19;8x20;8x21;8x22;8x23;8x24;8x9;9x11;9x12;9x13;9x14;9x15;9x16;9x17;9x18;9x19;9x20;9x21;9x22;9x23;9x24;9x8;9x9`
- output_shapes: `10x10;10x11;10x12;10x13;10x14;10x15;10x16;10x17;10x18;10x19;10x20;10x21;10x22;10x24;10x7;10x8;11x10;11x7;11x8;11x9;12x10;12x7;12x8;12x9;13x10;13x7;13x8;14x10;14x7;14x8;15x10;15x7;15x9;16x10;16x7;16x8;16x9;17x10;17x7;17x8;17x9;18x10;18x8;18x9;19x10;19x7;19x8;19x9;20x10;20x7;20x8;20x9;21x7;21x8;21x9;22x10;22x7;22x8;22x9;23x10;23x8;7x10;7x11;7x12;7x13;7x14;7x15;7x16;7x17;7x18;7x19;7x20;7x21;7x22;7x24;7x9;8x10;8x11;8x12;8x13;8x14;8x15;8x16;8x18;8x19;8x20;8x21;8x22;8x23;8x24;8x9;9x11;9x12;9x13;9x14;9x15;9x16;9x17;9x18;9x19;9x20;9x21;9x22;9x23;9x24;9x8;9x9`
- same_shape_all_examples: `True`
- output_colors_subset_input: `True`
- avg_changed_cell_ratio_same_shape: `0.1639`

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

1. `python "workplace C/neurogolf-2026-work/scripts/c_score_scan_artifacts.py" --tasks task284 --score-top-n 8 --full-validate`
2. Inspect best lower-size artifacts in `workplace C/score_docs/artifact_scans/` and compare memory/params split for `task284`.
3. If a lower-cost artifact validates, register it with `c_cost_diff_runner.py --task task284 --old-artifact <current> --new-artifact <candidate> --accept-if-better`.

## Cost Diff

| attempt | old_cost | new_cost | delta_cost | local_valid | accepted | artifact_path |
| --- | ---: | ---: | ---: | --- | --- | --- |
| pending |  |  |  |  |  |  |

## Attempts

- No accepted C-local attempt recorded yet.

## Next Best Action

- Run artifact scan and accept the first full-validation lower-cost artifact.
