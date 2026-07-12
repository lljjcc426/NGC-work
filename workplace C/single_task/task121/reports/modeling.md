# task121 independent modeling

## Rule

The 13x13 input contains several 3x3 objects. One object contains a single color-8 marker. The output is that marked object's 3x3 occupancy mask, recolored uniformly with the object's nonzero, non-8 body color; empty cells become color 0.

## Candidate structure

`task121_marker_object.onnx` locates the marker row and column with coordinate-weighted Einsums, slices the centered 3x3 patch, derives object occupancy from the background channel, obtains the body color by spatially reducing the patch while excluding channels 0 and 8, then reconstructs the output with a single Where. This is an object-extraction model independent of the baseline's top-row offset heuristic.

An initial NonZero locator was discarded because the official sanitizer forbids NonZero. The final coordinate-weighted locator uses permitted operators and has statically declared 3x3 intermediates so official memory scoring succeeds.

## Validation

- Public examples: 3 train, 1 test, 262 arc-gen.
- Candidate: 266/266 exact.
- Baseline cost: 326 (memory 260, params 66).
- Candidate cost: 4582 (memory 4517, params 65).
- Decision: rule accepted as an independent model; replacement rejected because the explicit object patch tensors are more expensive.
