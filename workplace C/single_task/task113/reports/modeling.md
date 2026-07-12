# task113 independent modeling

## Rule

The logical canvas always has ten rows and a variable width. The upper five rows form the source half; the lower half is the upper half in reverse row order. Empty rows in the source half remain part of the reflection. Equivalently, output row indices are `[0,1,2,3,4,4,3,2,1,0]`.

## Independent structure

`scripts/build_mirror_concat.py` does not use the baseline Gather table. It slices rows 0 through 4, reverses that tensor with a negative-step Slice, concatenates source and reflection on the height axis, then pads the ten-row result to the benchmark shape.

This directly models the geometric reflection. It is not an Identity, opset change, initializer rewrite, or audit-only result.

## Official validation and cost

| variant | passed | checked | memory | params | cost |
|---|---:|---:|---:|---:|---:|
| baseline Gather | 265 | 265 | 0 | 30 | 30 |
| Slice/reverse/Concat | 265 | 265 | 24000 | 14 | 24014 |

The candidate reduces stored parameters but materializes wide intermediate tensors. It is mathematically exact and structurally independent, but not a score replacement.
