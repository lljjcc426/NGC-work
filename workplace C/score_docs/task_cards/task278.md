# task278 Score Card

Generated: 2026-07-09T15:34:05

## Score Priority

- priority_band: `P1_16_16p7`
- assignment_cost: `5010`
- assignment_points: `16.480809`
- current_cost: `4505.0`
- current_score: `16.58705682995756`
- quick depth: `deep P0/P1`

## Why This Task Matters

- C track role: `onnx_equiv_compression`.
- Score gap focus: cost compression on current ONNX artifact; no rule rewrite is accepted without full validation.

## Structure

- shape_class: `same_shape`
- size_trend: `same_size`
- color_class: `new_output_colors`
- train/test/arc-gen: `2/1/262`
- input_shapes: `15x15;15x16;15x17;15x18;16x15;16x16;16x17;16x18;17x15;17x16;17x17;17x18;18x15;18x16;18x17;18x18`
- output_shapes: `15x15;15x16;15x17;15x18;16x15;16x16;16x17;16x18;17x15;17x16;17x17;17x18;18x15;18x16;18x17;18x18`
- same_shape_all_examples: `True`
- output_colors_subset_input: `False`
- avg_changed_cell_ratio_same_shape: `0.1324`

## Pattern Understanding

- Same-shape task. Prioritize mask/color logic compression and removal of redundant full-grid branches.
- Output introduces colors not always present in input; recolor constants are risk-sensitive.

## ONNX Compression Opportunities

- Run artifact scan against all local public and candidate ONNX sources.
- Fully validate any lower-cost artifact on train + test + arc-gen before accepting.
- Compare official memory/params split to locate whether memory graph or constants dominate.
- Try same-shape fast path: simplify masks, color comparisons, and identity-preserving branches.
- Audit recolor constants and replace broad per-color logic with narrow constant/color-map paths.
- Changed-cell ratio is low; test sparse mask overlay instead of full-grid recompute.

## Concrete Next Experiments

1. `python "workplace C/neurogolf-2026-work/scripts/c_score_scan_artifacts.py" --tasks task278 --score-top-n 8 --full-validate`
2. Inspect best lower-size artifacts in `workplace C/score_docs/artifact_scans/` and compare memory/params split for `task278`.
3. If a lower-cost artifact validates, register it with `c_cost_diff_runner.py --task task278 --old-artifact <current> --new-artifact <candidate> --accept-if-better`.

## Cost Diff

| attempt | old_cost | new_cost | delta_cost | local_valid | accepted | artifact_path |
| --- | ---: | ---: | ---: | --- | --- | --- |
| pending |  |  |  |  |  |  |

## Attempts

- No accepted C-local attempt recorded yet.

## Next Best Action

- Run artifact scan and accept the first full-validation lower-cost artifact.
