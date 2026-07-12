# task351 independent modeling

## Rule

All 265 public examples use a 16x16 logical input and a 5x5 output. The input is rotationally symmetric except for one 5x5 payload. Color 3 forms the locator: the row and column with the largest color-3 counts identify the opposite corner of the payload. Starting from `[15-row, 15-col]`, reading five cells upward and leftward produces the answer. Channel 0 is excluded from the extracted patch and restored by a leading channel pad.

## Structural candidate

`scripts/build_reduce_gather_locator.py` replaces both baseline `Einsum` locator reductions with:

1. `ReduceSum` over width/height for every color.
2. `Gather` channel 3 and `Squeeze` the retained channel axis.
3. `ArgMax` for the marker row and column.
4. Dynamic reverse `Slice` for the 5x5 payload and `Pad` to the benchmark tensor.

This is a different locator dataflow, not an opset or initializer-only edit. The dynamic slice has explicit `[1,9,5,5]` value information so the official memory scorer can score it.

## Result

- Baseline: 265/265, cost 1229.
- Candidate: 265/265, cost 3863.
- Decision: reject for replacement. Parameters fall from 29 to 23, but the two all-color reduction tensors raise memory from 1200 to 3840. A competitive follow-up must preserve channel selection before reduction, as the baseline Einsum does.
