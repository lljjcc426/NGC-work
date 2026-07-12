# task362 independent modeling

## Rule

The input contains a full 10x10 cross in one nonzero, non-5 color and a vertical run of color-5 markers at the right edge. If the marker count is `k`, the cross intersection moves from `(r,c)` to `(r+k,c-k)`, and full horizontal and vertical lines are redrawn through that new intersection.

## Candidate structure

`task362_dynamic_shift.onnx` identifies the cross color from its count of 19, computes the original intersection analytically from row/column moments of the 19-cell cross, applies the marker-count offset, compares the target coordinates with fixed 0-9 coordinate vectors, and reconstructs a full cross. This is a moment-based geometric model and does not use the baseline ScatterElements term construction.

An initial literal translation model failed because the rule redraws full-length lines rather than shifting and clipping the original bitmap. That failure led to the final intersection-and-redraw model.

## Validation

- Public examples: 4 train, 1 test, 262 arc-gen.
- Final candidate: 267/267 exact.
- Baseline cost: 521 (memory 408, params 113).
- Candidate cost: 18041 (memory 17974, params 67).
- Decision: geometric rule accepted as an independent model; replacement rejected because explicit 10x10 one-hot reconstruction dominates memory.
