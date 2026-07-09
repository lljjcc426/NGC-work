# task178 Score Card

Generated: 2026-07-09T15:34:05

## Score Priority

- priority_band: `P3_ge17p5`
- assignment_cost: `762`
- assignment_points: `18.364053`
- current_cost: `762.0`
- current_score: `18.364053444313353`
- quick depth: `light P2/P3`

## Why This Task Matters

- C track role: `onnx_equiv_compression`.
- Score gap focus: cost compression on current ONNX artifact; no rule rewrite is accepted without full validation.

## Structure

- shape_class: `shape_change`
- size_trend: `shrink`
- color_class: `input_palette_only`
- train/test/arc-gen: `5/1/262`
- input_shapes: `10x1;10x2;10x3;10x4;10x5;11x1;11x2;11x3;11x4;11x5;12x1;12x3;12x4;14x2;1x10;1x11;1x13;1x4;1x5;1x6;1x7;1x8;1x9;2x10;2x11;2x12;2x5;2x6;2x7;2x8;2x9;3x10;3x11;3x12;3x13;3x3;3x5;3x6;3x7;3x8;3x9;4x1;4x10;4x11;4x12;4x13;4x2;4x3;4x4;4x5;4x6;4x7;4x8;4x9;5x10;5x11;5x12;5x2;5x3;5x4;5x5;5x6;5x7;5x8;5x9;6x1;6x2;6x3;6x4;6x5;7x1;7x2;7x3;7x4;7x5;8x1;8x2;8x3;8x4;8x5;9x1;9x2;9x3;9x4;9x5`
- output_shapes: `1x3;1x4;1x5;3x1;4x1;5x1`
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

1. `python "workplace C/neurogolf-2026-work/scripts/c_score_scan_artifacts.py" --tasks task178 --score-top-n 8 --full-validate`
2. Inspect best lower-size artifacts in `workplace C/score_docs/artifact_scans/` and compare memory/params split for `task178`.
3. If a lower-cost artifact validates, register it with `c_cost_diff_runner.py --task task178 --old-artifact <current> --new-artifact <candidate> --accept-if-better`.

## Cost Diff

| attempt | old_cost | new_cost | delta_cost | local_valid | accepted | artifact_path |
| --- | ---: | ---: | ---: | --- | --- | --- |
| pending |  |  |  |  |  |  |

## Attempts

- No accepted C-local attempt recorded yet.

## Next Best Action

- Run artifact scan and accept the first full-validation lower-cost artifact.
