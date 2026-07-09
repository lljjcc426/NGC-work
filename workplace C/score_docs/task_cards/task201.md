# task201 Score Card

Generated: 2026-07-09T15:34:05

## Score Priority

- priority_band: `P2_16p7_17p5`
- assignment_cost: `3349`
- assignment_points: `16.883583`
- current_cost: `3046.0`
- current_score: `16.978415466544895`
- quick depth: `light P2/P3`

## Why This Task Matters

- C track role: `onnx_equiv_compression`.
- Score gap focus: cost compression on current ONNX artifact; no rule rewrite is accepted without full validation.

## Structure

- shape_class: `shape_change`
- size_trend: `shrink`
- color_class: `input_palette_only`
- train/test/arc-gen: `4/1/261`
- input_shapes: `13x13`
- output_shapes: `4x6;4x8;5x6;5x8;6x6;6x8;7x8`
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

1. `python "workplace C/neurogolf-2026-work/scripts/c_score_scan_artifacts.py" --tasks task201 --score-top-n 8 --full-validate`
2. Inspect best lower-size artifacts in `workplace C/score_docs/artifact_scans/` and compare memory/params split for `task201`.
3. If a lower-cost artifact validates, register it with `c_cost_diff_runner.py --task task201 --old-artifact <current> --new-artifact <candidate> --accept-if-better`.

## Cost Diff

| attempt | old_cost | new_cost | delta_cost | local_valid | accepted | artifact_path |
| --- | ---: | ---: | ---: | --- | --- | --- |
| pending |  |  |  |  |  |  |

## Attempts

- No accepted C-local attempt recorded yet.

## Next Best Action

- Run artifact scan and accept the first full-validation lower-cost artifact.
