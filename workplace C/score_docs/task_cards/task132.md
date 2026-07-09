# task132 Score Card

Generated: 2026-07-09T15:34:05

## Score Priority

- priority_band: `P1_16_16p7`
- assignment_cost: `4089`
- assignment_points: `16.683944`
- current_cost: `3652.0`
- current_score: `16.79696975828514`
- quick depth: `deep P0/P1`

## Why This Task Matters

- C track role: `onnx_equiv_compression`.
- Score gap focus: cost compression on current ONNX artifact; no rule rewrite is accepted without full validation.

## Structure

- shape_class: `same_shape`
- size_trend: `same_size`
- color_class: `input_palette_only`
- train/test/arc-gen: `4/1/262`
- input_shapes: `10x10;10x11;10x12;10x13;10x14;10x15;10x6;10x7;10x8;10x9;11x10;11x11;11x12;11x13;11x15;11x6;11x7;11x8;12x10;12x11;12x12;12x13;12x14;12x6;12x7;12x8;12x9;13x10;13x11;13x12;13x13;13x14;13x15;13x6;13x7;13x8;13x9;14x10;14x11;14x12;14x13;14x14;14x15;14x7;14x8;14x9;15x10;15x11;15x12;15x13;15x14;15x15;15x6;15x7;15x8;15x9;6x10;6x11;6x12;6x13;6x14;6x15;6x6;6x7;6x8;6x9;7x10;7x11;7x12;7x13;7x14;7x15;7x6;7x7;7x8;7x9;8x10;8x11;8x12;8x14;8x15;8x6;8x7;8x8;9x10;9x11;9x12;9x13;9x14;9x6;9x7;9x8;9x9`
- output_shapes: `10x10;10x11;10x12;10x13;10x14;10x15;10x6;10x7;10x8;10x9;11x10;11x11;11x12;11x13;11x15;11x6;11x7;11x8;12x10;12x11;12x12;12x13;12x14;12x6;12x7;12x8;12x9;13x10;13x11;13x12;13x13;13x14;13x15;13x6;13x7;13x8;13x9;14x10;14x11;14x12;14x13;14x14;14x15;14x7;14x8;14x9;15x10;15x11;15x12;15x13;15x14;15x15;15x6;15x7;15x8;15x9;6x10;6x11;6x12;6x13;6x14;6x15;6x6;6x7;6x8;6x9;7x10;7x11;7x12;7x13;7x14;7x15;7x6;7x7;7x8;7x9;8x10;8x11;8x12;8x14;8x15;8x6;8x7;8x8;9x10;9x11;9x12;9x13;9x14;9x6;9x7;9x8;9x9`
- same_shape_all_examples: `True`
- output_colors_subset_input: `True`
- avg_changed_cell_ratio_same_shape: `0.2850`

## Pattern Understanding

- Same-shape task. Prioritize mask/color logic compression and removal of redundant full-grid branches.
- Output palette is input-contained; unused color creation branches are likely removable.

## ONNX Compression Opportunities

- Run artifact scan against all local public and candidate ONNX sources.
- Fully validate any lower-cost artifact on train + test + arc-gen before accepting.
- Compare official memory/params split to locate whether memory graph or constants dominate.
- Try same-shape fast path: simplify masks, color comparisons, and identity-preserving branches.
- Exploit input-palette-only constraint; remove unused output color branches and redundant compares.

## Concrete Next Experiments

1. `python "workplace C/neurogolf-2026-work/scripts/c_score_scan_artifacts.py" --tasks task132 --score-top-n 8 --full-validate`
2. Inspect best lower-size artifacts in `workplace C/score_docs/artifact_scans/` and compare memory/params split for `task132`.
3. If a lower-cost artifact validates, register it with `c_cost_diff_runner.py --task task132 --old-artifact <current> --new-artifact <candidate> --accept-if-better`.

## Cost Diff

| attempt | old_cost | new_cost | delta_cost | local_valid | accepted | artifact_path |
| --- | ---: | ---: | ---: | --- | --- | --- |
| pending |  |  |  |  |  |  |

## Attempts

- No accepted C-local attempt recorded yet.

## Next Best Action

- Run artifact scan and accept the first full-validation lower-cost artifact.
