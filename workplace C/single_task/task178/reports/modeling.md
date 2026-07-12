# task178 independent modeling

## Rule

The input consists of repeated horizontal or vertical bands. Select the axis whose first line changes between adjacent cells, then run-length encode that line: emit the first color and each later nonzero color whose value differs from its predecessor. A vertical band input produces a column; a horizontal band input produces a row. Public outputs contain at most five runs.

## Structural candidate

`scripts/build_argmax_compress.py` replaces the baseline's two 1x1 label convolutions with a full-grid `ArgMax` over color channels. It extracts the first row and column, detects the active orientation, builds run starts, and uses permitted `TopK + Gather` to retain the first five runs. An initial `Compress` design was semantically exact but rejected because NeuroGolf prohibits that operator; the saved candidate is the permitted TopK version and was rescored from scratch.

## Result

- Baseline: 268/268, cost 762.
- Candidate: 268/268, cost 73884.
- Decision: reject for replacement. The independent ArgMax model is exact, but its `30x30` label map dominates memory. The baseline's strided 1x1 convolutions read only the first row/column and are far cheaper.
