# task307 Score Card

Generated: 2026-07-09T15:34:05

## Score Priority

- priority_band: `P3_ge17p5`
- assignment_cost: `5`
- assignment_points: `23.390562`
- current_cost: `5.0`
- current_score: `23.3905620875659`
- quick depth: `light P2/P3`

## Why This Task Matters

- C track role: `onnx_equiv_compression`.
- Score gap focus: cost compression on current ONNX artifact; no rule rewrite is accepted without full validation.

## Structure

- shape_class: `shape_change`
- size_trend: `grow`
- color_class: `input_palette_only`
- train/test/arc-gen: `3/1/262`
- input_shapes: `2x2;3x3;4x4;5x5`
- output_shapes: `10x10;4x4;6x6;8x8`
- same_shape_all_examples: `False`
- output_colors_subset_input: `True`
- avg_changed_cell_ratio_same_shape: `0.0000`

## Pattern Understanding

- Grow task. Prioritize shape construction and repeat/tile path review before graph-level simplification.
- Output palette is input-contained; unused color creation branches are likely removable.

## ONNX Compression Opportunities

- Run artifact scan against all local public and candidate ONNX sources.
- Fully validate any lower-cost artifact on train + test + arc-gen before accepting.
- Compare official memory/params split to locate whether memory graph or constants dominate.
- Trace output shape path first; shape-change tasks need crop/grow logic preservation before compression.
- Exploit input-palette-only constraint; remove unused output color branches and redundant compares.

## Concrete Next Experiments

1. `python "workplace C/neurogolf-2026-work/scripts/c_score_scan_artifacts.py" --tasks task307 --score-top-n 8 --full-validate`
2. Inspect best lower-size artifacts in `workplace C/score_docs/artifact_scans/` and compare memory/params split for `task307`.
3. If a lower-cost artifact validates, register it with `c_cost_diff_runner.py --task task307 --old-artifact <current> --new-artifact <candidate> --accept-if-better`.

## Cost Diff

| attempt | old_cost | new_cost | delta_cost | local_valid | accepted | artifact_path |
| --- | ---: | ---: | ---: | --- | --- | --- |
| pending |  |  |  |  |  |  |

## Attempts

- No accepted C-local attempt recorded yet.

## Next Best Action

- Run artifact scan and accept the first full-validation lower-cost artifact.
