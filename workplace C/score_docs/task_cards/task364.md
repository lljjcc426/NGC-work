# task364 Score Card

Generated: 2026-07-09T15:34:05

## Score Priority

- priority_band: `P0_lt16`
- assignment_cost: `20802`
- assignment_points: `15.057196`
- current_cost: `14642.0`
- current_score: `15.408350609793413`
- quick depth: `deep P0/P1`

## Why This Task Matters

- C track role: `onnx_equiv_compression`.
- Score gap focus: cost compression on current ONNX artifact; no rule rewrite is accepted without full validation.

## Structure

- shape_class: `same_shape`
- size_trend: `same_size`
- color_class: `new_output_colors`
- train/test/arc-gen: `3/1/262`
- input_shapes: `10x10;10x11;10x12;10x8;10x9;11x10;11x11;11x12;11x13;11x9;12x10;12x11;12x12;12x13;12x14;13x11;13x12;13x13;13x14;13x15;14x12;14x13;14x14;14x15;14x16;15x13;15x14;15x15;15x16;15x17;16x14;16x15;16x16;16x17;16x18;17x15;17x16;17x17;17x18;17x19;18x16;18x17;18x18;18x19;18x20;19x17;19x18;19x19;19x20;19x21;20x18;20x19;20x20;20x21;20x22`
- output_shapes: `10x10;10x11;10x12;10x8;10x9;11x10;11x11;11x12;11x13;11x9;12x10;12x11;12x12;12x13;12x14;13x11;13x12;13x13;13x14;13x15;14x12;14x13;14x14;14x15;14x16;15x13;15x14;15x15;15x16;15x17;16x14;16x15;16x16;16x17;16x18;17x15;17x16;17x17;17x18;17x19;18x16;18x17;18x18;18x19;18x20;19x17;19x18;19x19;19x20;19x21;20x18;20x19;20x20;20x21;20x22`
- same_shape_all_examples: `True`
- output_colors_subset_input: `False`
- avg_changed_cell_ratio_same_shape: `0.2165`

## Pattern Understanding

- Same-shape task. Prioritize mask/color logic compression and removal of redundant full-grid branches.
- Output introduces colors not always present in input; recolor constants are risk-sensitive.

## ONNX Compression Opportunities

- Run artifact scan against all local public and candidate ONNX sources.
- Fully validate any lower-cost artifact on train + test + arc-gen before accepting.
- Compare official memory/params split to locate whether memory graph or constants dominate.
- Try same-shape fast path: simplify masks, color comparisons, and identity-preserving branches.
- Audit recolor constants and replace broad per-color logic with narrow constant/color-map paths.

## Concrete Next Experiments

1. `python "workplace C/neurogolf-2026-work/scripts/c_score_scan_artifacts.py" --tasks task364 --score-top-n 8 --full-validate`
2. Inspect best lower-size artifacts in `workplace C/score_docs/artifact_scans/` and compare memory/params split for `task364`.
3. If a lower-cost artifact validates, register it with `c_cost_diff_runner.py --task task364 --old-artifact <current> --new-artifact <candidate> --accept-if-better`.

## Cost Diff

| attempt | old_cost | new_cost | delta_cost | local_valid | accepted | artifact_path |
| --- | ---: | ---: | ---: | --- | --- | --- |
| pending |  |  |  |  |  |  |

## Attempts

- No accepted C-local attempt recorded yet.

## Next Best Action

- Run artifact scan and accept the first full-validation lower-cost artifact.
