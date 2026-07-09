# task159 Score Card

Generated: 2026-07-09T15:34:05

## Score Priority

- priority_band: `P3_ge17p5`
- assignment_cost: `1568`
- assignment_points: `17.642444`
- current_cost: `1568.0`
- current_score: `17.642443799089648`
- quick depth: `light P2/P3`

## Why This Task Matters

- C track role: `onnx_equiv_compression`.
- Score gap focus: cost compression on current ONNX artifact; no rule rewrite is accepted without full validation.

## Structure

- shape_class: `shape_change`
- size_trend: `shrink`
- color_class: `input_palette_only`
- train/test/arc-gen: `3/1/261`
- input_shapes: `15x16;15x17;15x18;15x19;15x20;15x21;15x23;15x24;15x27;16x15;16x19;16x20;16x21;16x22;16x23;16x25;16x26;16x27;16x28;16x29;16x30;17x15;17x16;17x17;17x18;17x20;17x21;17x23;17x24;17x27;17x28;18x15;18x16;18x17;18x18;18x19;18x20;18x21;18x22;18x23;18x24;18x25;18x26;18x27;18x29;18x30;19x15;19x17;19x22;19x24;19x25;19x27;19x28;19x29;20x15;20x17;20x19;20x20;20x23;20x24;20x25;20x26;20x27;21x15;21x16;21x17;21x18;21x20;21x21;21x22;21x23;21x24;21x25;21x26;21x27;21x28;21x29;21x30;22x15;22x17;22x18;22x19;22x23;22x25;22x26;22x27;22x29;22x30;23x15;23x16;23x17;23x18;23x20;23x21;23x24;23x26;23x28;23x29;24x16;24x18;24x21;24x22;24x23;24x26;24x28;24x29;25x16;25x17;25x18;25x19;25x20;25x23;25x24;25x26;25x27;25x29;26x15;26x16;26x17;26x18;26x20;26x22;26x23;26x24;26x26;26x27;26x28;27x16;27x17;27x22;27x24;27x25;27x26;27x28;27x29;28x16;28x17;28x18;28x19;28x20;28x21;28x22;28x23;28x24;28x27;28x28;28x29;29x15;29x16;29x17;29x18;29x23;29x24;29x25;29x27;29x30;30x17;30x18;30x19;30x21;30x22;30x23;30x24;30x26;30x29;30x30`
- output_shapes: `11x11;14x14;5x5;8x8`
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

1. `python "workplace C/neurogolf-2026-work/scripts/c_score_scan_artifacts.py" --tasks task159 --score-top-n 8 --full-validate`
2. Inspect best lower-size artifacts in `workplace C/score_docs/artifact_scans/` and compare memory/params split for `task159`.
3. If a lower-cost artifact validates, register it with `c_cost_diff_runner.py --task task159 --old-artifact <current> --new-artifact <candidate> --accept-if-better`.

## Cost Diff

| attempt | old_cost | new_cost | delta_cost | local_valid | accepted | artifact_path |
| --- | ---: | ---: | ---: | --- | --- | --- |
| pending |  |  |  |  |  |  |

## Attempts

- No accepted C-local attempt recorded yet.

## Next Best Action

- Run artifact scan and accept the first full-validation lower-cost artifact.
