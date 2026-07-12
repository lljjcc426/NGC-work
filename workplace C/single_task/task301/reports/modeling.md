# task301 independent modeling

## Rule

Each nonzero color is a single horizontal bar, and its cell count is its length. The output keeps the input canvas size, right-aligns every bar to the maximum bar width, and places a bar of length `k` on row `height - width + k - 1`. Thus short bars are above long bars and the longest bar is on the bottom row; unused cells inside the canvas are color 0.

## Structural candidate

`scripts/build_direct_bar_masks.py` derives per-color counts, width, height, and gap, then constructs the output through explicit broadcast masks:

- row condition: `row + 1 == gap + count[color]`;
- column condition: `col + 1 > width - count[color]`;
- canvas condition: `row < height` and `col < width`;
- channel 0 is the valid-canvas complement of all foreground masks.

This replaces the baseline threshold-table/`Greater` formulation with direct semantic masks. It is a genuine independent solver and does not reuse the baseline threshold constants.

## Result

- Baseline: 266/266, cost 1141.
- Candidate: 266/266, cost 36751.
- Decision: reject for replacement. The direct rule is exact, but materializing nine `30x30` color masks and their Boolean intermediates dominates memory. The baseline's threshold representation is substantially more compact.
