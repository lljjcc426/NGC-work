# task142 independent modeling

## Rule

The 3x3 input is expanded into a 6x6 square by reflection across its right and bottom edges:

```text
[ input          horizontal_flip(input) ]
[ vertical_flip  both_axes_flip(input)  ]
```

The edge cells are duplicated because the output consists of two complete three-cell halves on each axis.

## Independent structure

`scripts/build_mirror_quadrants.py` crops the logical patch, creates horizontal, vertical, and double reflections with negative-step Slice nodes, joins the four quadrants with Concat, and pads the result. This is a geometric decomposition of the transformation rather than the parent's six-input multiplicative Einsum.

## Official validation and cost

| variant | passed | checked | memory | params | cost |
|---|---:|---:|---:|---:|---:|
| factorized Einsum | 266 | 266 | 0 | 90 | 90 |
| Slice/Concat quadrants | 266 | 266 | 4320 | 25 | 4345 |

The candidate is exact and uses fewer constants, but each mirrored patch is a scored intermediate. It is not suitable for replacing the fused Einsum.
