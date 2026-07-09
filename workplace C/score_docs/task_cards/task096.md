# task096 Score Card

Generated: 2026-07-09T15:34:05

## Score Priority

- priority_band: `P1_16_16p7`
- assignment_cost: `7678`
- assignment_points: `16.053886`
- current_cost: `7678.0`
- current_score: `16.053885624439257`
- quick depth: `deep P0/P1`

## Why This Task Matters

- C track role: `onnx_equiv_compression`.
- Score gap focus: cost compression on current ONNX artifact; no rule rewrite is accepted without full validation.

## Structure

- shape_class: `shape_change`
- size_trend: `shrink`
- color_class: `input_palette_only`
- train/test/arc-gen: `3/1/262`
- input_shapes: `13x13;13x14;13x15;13x16;13x17;13x18;13x19;14x13;14x14;14x15;14x16;14x17;14x18;14x19;15x13;15x14;15x15;15x16;15x17;15x18;15x19;16x13;16x14;16x15;16x16;16x17;16x18;16x19;17x13;17x14;17x15;17x16;17x17;17x18;17x19;18x13;18x14;18x15;18x16;18x17;18x18;18x19;19x13;19x14;19x15;19x16;19x17;19x18;19x19`
- output_shapes: `11x11;7x7;9x9`
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

1. `python "workplace C/neurogolf-2026-work/scripts/c_score_scan_artifacts.py" --tasks task096 --score-top-n 8 --full-validate`
2. Inspect best lower-size artifacts in `workplace C/score_docs/artifact_scans/` and compare memory/params split for `task096`.
3. If a lower-cost artifact validates, register it with `c_cost_diff_runner.py --task task096 --old-artifact <current> --new-artifact <candidate> --accept-if-better`.

## Cost Diff

| attempt | old_cost | new_cost | delta_cost | local_valid | accepted | artifact_path |
| --- | ---: | ---: | ---: | --- | --- | --- |
| pending |  |  |  |  |  |  |

## Attempts

- No accepted C-local attempt recorded yet.

## Next Best Action

- Run artifact scan and accept the first full-validation lower-cost artifact.
