# task091 Score Card

Generated: 2026-07-09T15:34:05

## Score Priority

- priority_band: `P2_16p7_17p5`
- assignment_cost: `3013`
- assignment_points: `16.989308`
- current_cost: `2764.0`
- current_score: `17.07556581511244`
- quick depth: `light P2/P3`

## Why This Task Matters

- C track role: `onnx_equiv_compression`.
- Score gap focus: cost compression on current ONNX artifact; no rule rewrite is accepted without full validation.

## Structure

- shape_class: `shape_change`
- size_trend: `shrink`
- color_class: `input_palette_only`
- train/test/arc-gen: `3/1/262`
- input_shapes: `10x10;10x11;10x12;10x13;10x14;10x15;10x9;11x10;11x11;11x12;11x13;11x14;11x15;11x9;12x10;12x11;12x12;12x13;12x14;12x15;12x9;13x10;13x11;13x12;13x13;13x14;13x15;13x9;14x10;14x11;14x12;14x13;14x14;14x15;14x9;15x10;15x11;15x12;15x13;15x14;15x15;15x9;9x10;9x11;9x12;9x13;9x14;9x15;9x9`
- output_shapes: `10x11;10x13;10x3;10x4;10x5;10x6;10x7;10x8;10x9;11x10;11x11;11x12;11x3;11x5;11x6;11x7;12x12;12x5;12x6;12x7;12x8;12x9;13x13;13x5;13x7;13x9;14x6;14x7;14x8;3x10;3x11;3x12;3x13;3x3;3x4;3x5;3x6;3x7;3x8;3x9;4x10;4x11;4x12;4x13;4x3;4x4;4x5;4x6;4x7;4x8;4x9;5x10;5x11;5x12;5x14;5x3;5x4;5x5;5x6;5x7;5x8;6x10;6x12;6x3;6x4;6x5;6x6;6x7;6x8;6x9;7x10;7x11;7x3;7x4;7x5;7x6;7x7;7x8;7x9;8x10;8x11;8x12;8x3;8x4;8x5;8x6;8x7;8x8;8x9;9x10;9x12;9x13;9x3;9x4;9x5;9x6;9x7;9x8;9x9`
- same_shape_all_examples: `False`
- output_colors_subset_input: `True`
- avg_changed_cell_ratio_same_shape: `0.0000`

## Pattern Understanding

- Shrink task. Prioritize crop/bounding-box/selection path review before graph-level simplification.
- Output palette is input-contained; unused color creation branches are likely removable.

## ONNX Compression Opportunities

- Run artifact scan against all local public and candidate ONNX sources.
- Fully validate any lower-cost artifact on train + test + arc-gen before accepting.
- Compare official memory/params split to locate whether memory graph or constants dominate.
- Trace output shape path first; shape-change tasks need crop/grow logic preservation before compression.
- Exploit input-palette-only constraint; remove unused output color branches and redundant compares.

## Concrete Next Experiments

1. `python "workplace C/neurogolf-2026-work/scripts/c_score_scan_artifacts.py" --tasks task091 --score-top-n 8 --full-validate`
2. Inspect best lower-size artifacts in `workplace C/score_docs/artifact_scans/` and compare memory/params split for `task091`.
3. If a lower-cost artifact validates, register it with `c_cost_diff_runner.py --task task091 --old-artifact <current> --new-artifact <candidate> --accept-if-better`.

## Cost Diff

| attempt | old_cost | new_cost | delta_cost | local_valid | accepted | artifact_path |
| --- | ---: | ---: | ---: | --- | --- | --- |
| pending |  |  |  |  |  |  |

## Attempts

- No accepted C-local attempt recorded yet.

## Next Best Action

- Run artifact scan and accept the first full-validation lower-cost artifact.
