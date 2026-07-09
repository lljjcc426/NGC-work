# task388 Score Card

Generated: 2026-07-09T15:34:05

## Score Priority

- priority_band: `P2_16p7_17p5`
- assignment_cost: `2190`
- assignment_points: `17.308343`
- current_cost: `1829.0`
- current_score: `17.488475351609132`
- quick depth: `light P2/P3`

## Why This Task Matters

- C track role: `onnx_equiv_compression`.
- Score gap focus: cost compression on current ONNX artifact; no rule rewrite is accepted without full validation.

## Structure

- shape_class: `shape_change`
- size_trend: `grow`
- color_class: `new_output_colors`
- train/test/arc-gen: `3/1/262`
- input_shapes: `2x2;3x3;4x4;5x5;6x6`
- output_shapes: `10x10;12x12;4x4;6x6;8x8`
- same_shape_all_examples: `False`
- output_colors_subset_input: `False`
- avg_changed_cell_ratio_same_shape: `0.0000`

## Pattern Understanding

- Grow task. Prioritize shape construction and repeat/tile path review before graph-level simplification.
- Output introduces colors not always present in input; recolor constants are risk-sensitive.

## ONNX Compression Opportunities

- Run artifact scan against all local public and candidate ONNX sources.
- Fully validate any lower-cost artifact on train + test + arc-gen before accepting.
- Compare official memory/params split to locate whether memory graph or constants dominate.
- Trace output shape path first; shape-change tasks need crop/grow logic preservation before compression.
- Audit recolor constants and replace broad per-color logic with narrow constant/color-map paths.

## Concrete Next Experiments

1. `python "workplace C/neurogolf-2026-work/scripts/c_score_scan_artifacts.py" --tasks task388 --score-top-n 8 --full-validate`
2. Inspect best lower-size artifacts in `workplace C/score_docs/artifact_scans/` and compare memory/params split for `task388`.
3. If a lower-cost artifact validates, register it with `c_cost_diff_runner.py --task task388 --old-artifact <current> --new-artifact <candidate> --accept-if-better`.

## Cost Diff

| attempt | old_cost | new_cost | delta_cost | local_valid | accepted | artifact_path |
| --- | ---: | ---: | ---: | --- | --- | --- |
| pending |  |  |  |  |  |  |

## Attempts

- No accepted C-local attempt recorded yet.

## Next Best Action

- Run artifact scan and accept the first full-validation lower-cost artifact.
