# task203 Score Card

Generated: 2026-07-09T15:34:05

## Score Priority

- priority_band: `P3_ge17p5`
- assignment_cost: `355`
- assignment_points: `19.127882`
- current_cost: `355.0`
- current_score: `19.127882210524582`
- quick depth: `light P2/P3`

## Why This Task Matters

- C track role: `onnx_equiv_compression`.
- Score gap focus: cost compression on current ONNX artifact; no rule rewrite is accepted without full validation.

## Structure

- shape_class: `same_shape`
- size_trend: `same_size`
- color_class: `input_palette_only`
- train/test/arc-gen: `4/1/262`
- input_shapes: `10x10;12x12;14x14;16x16;18x18;6x6;8x8`
- output_shapes: `10x10;12x12;14x14;16x16;18x18;6x6;8x8`
- same_shape_all_examples: `True`
- output_colors_subset_input: `True`
- avg_changed_cell_ratio_same_shape: `0.8810`

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

1. `python "workplace C/neurogolf-2026-work/scripts/c_score_scan_artifacts.py" --tasks task203 --score-top-n 8 --full-validate`
2. Inspect best lower-size artifacts in `workplace C/score_docs/artifact_scans/` and compare memory/params split for `task203`.
3. If a lower-cost artifact validates, register it with `c_cost_diff_runner.py --task task203 --old-artifact <current> --new-artifact <candidate> --accept-if-better`.

## Cost Diff

| attempt | old_cost | new_cost | delta_cost | local_valid | accepted | artifact_path |
| --- | ---: | ---: | ---: | --- | --- | --- |
| pending |  |  |  |  |  |  |

## Attempts

- No accepted C-local attempt recorded yet.

## Next Best Action

- Run artifact scan and accept the first full-validation lower-cost artifact.
