# task335 Score Card

Generated: 2026-07-09T15:34:05

## Score Priority

- priority_band: `P2_16p7_17p5`
- assignment_cost: `3133`
- assignment_points: `16.950254`
- current_cost: `1380.0`
- current_score: `17.77016122184875`
- quick depth: `light P2/P3`

## Why This Task Matters

- C track role: `onnx_equiv_compression`.
- Score gap focus: cost compression on current ONNX artifact; no rule rewrite is accepted without full validation.

## Structure

- shape_class: `same_shape`
- size_trend: `same_size`
- color_class: `new_output_colors`
- train/test/arc-gen: `3/1/262`
- input_shapes: `10x10;10x11;10x12;10x14;10x16;10x17;10x18;10x19;10x20;11x10;11x11;11x13;11x14;11x15;11x16;11x17;11x18;11x19;11x20;12x10;12x11;12x12;12x13;12x14;12x15;12x16;12x18;12x19;12x20;13x10;13x11;13x12;13x13;13x15;13x16;13x17;13x18;13x20;14x10;14x11;14x12;14x13;14x14;14x15;14x16;14x17;14x18;14x19;14x20;15x10;15x11;15x12;15x14;15x15;15x16;15x17;15x19;15x20;16x10;16x12;16x13;16x14;16x15;16x16;16x17;16x18;16x20;17x10;17x11;17x12;17x13;17x14;17x15;17x16;17x17;17x18;17x19;18x11;18x12;18x13;18x14;18x15;18x16;18x17;18x18;18x19;19x10;19x11;19x12;19x13;19x14;19x15;19x17;19x18;19x19;19x20;20x10;20x11;20x12;20x14;20x15;20x16;20x18;20x20;8x11`
- output_shapes: `10x10;10x11;10x12;10x14;10x16;10x17;10x18;10x19;10x20;11x10;11x11;11x13;11x14;11x15;11x16;11x17;11x18;11x19;11x20;12x10;12x11;12x12;12x13;12x14;12x15;12x16;12x18;12x19;12x20;13x10;13x11;13x12;13x13;13x15;13x16;13x17;13x18;13x20;14x10;14x11;14x12;14x13;14x14;14x15;14x16;14x17;14x18;14x19;14x20;15x10;15x11;15x12;15x14;15x15;15x16;15x17;15x19;15x20;16x10;16x12;16x13;16x14;16x15;16x16;16x17;16x18;16x20;17x10;17x11;17x12;17x13;17x14;17x15;17x16;17x17;17x18;17x19;18x11;18x12;18x13;18x14;18x15;18x16;18x17;18x18;18x19;19x10;19x11;19x12;19x13;19x14;19x15;19x17;19x18;19x19;19x20;20x10;20x11;20x12;20x14;20x15;20x16;20x18;20x20;8x11`
- same_shape_all_examples: `True`
- output_colors_subset_input: `False`
- avg_changed_cell_ratio_same_shape: `0.0376`

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

1. `python "workplace C/neurogolf-2026-work/scripts/c_score_scan_artifacts.py" --tasks task335 --score-top-n 8 --full-validate`
2. Inspect best lower-size artifacts in `workplace C/score_docs/artifact_scans/` and compare memory/params split for `task335`.
3. If a lower-cost artifact validates, register it with `c_cost_diff_runner.py --task task335 --old-artifact <current> --new-artifact <candidate> --accept-if-better`.

## Cost Diff

| attempt | old_cost | new_cost | delta_cost | local_valid | accepted | artifact_path |
| --- | ---: | ---: | ---: | --- | --- | --- |
| pending |  |  |  |  |  |  |

## Attempts

- No accepted C-local attempt recorded yet.

## Next Best Action

- Run artifact scan and accept the first full-validation lower-cost artifact.
