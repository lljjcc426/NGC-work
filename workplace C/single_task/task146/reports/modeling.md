# task146 independent modeling

## Rule

The 9x3 input contains three vertically stacked 3x3 tiles. Exactly two tiles are symmetric across the main diagonal. The output is the unique tile that is not equal to its transpose.

## Candidate structure

`task146_asymmetry.onnx` crops the 9x3 region, gathers the three upper-triangle/lower-triangle cell pairs of every tile, compares their complete one-hot vectors, reduces the comparisons to one symmetry flag per tile, chooses the unique asymmetric tile, and gathers its three rows. This explicitly implements the symmetry rule rather than the baseline's weighted checksum Conv.

## Validation

- Public examples: 4 train, 1 test, 262 arc-gen.
- Candidate: 267/267 exact.
- Baseline cost: 265 (memory 165, params 100).
- Candidate cost: 7765 (memory 7717, params 48).
- Decision: rule accepted as an independent model; replacement rejected because Gather/Reshape comparison activations dominate cost.

## Round 2: low-width exact checksum

The second candidate avoids full-grid Gather. It factorizes the baseline's
`10x3x3` color-position checksum into two local convolutions:

1. A `1x1` Conv crops the `9x3` active area and maps the ten one-hot color
   channels to scalar color ids.
2. A `3x3` Conv computes the three transpose-pair differences with coefficients
   `(1, 2, 4)` and emits one checksum per tile.
3. The validated baseline row-selection tail is retained unchanged.

The coefficient search covered `(1, b, c)` for `1 <= b,c <= 32`. Across the
127 unique nonzero transpose-difference vectors in all 267 public examples,
`(1, 2, 4)` is the lexicographically smallest minimum-width choice with zero
checksum collisions. Detailed counts are in `checksum_collision_search.csv`.

Official `c_score_common.score_onnx(..., validate_all=True)` results:

| model | passed | memory | params | cost | points |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 267/267 | 165 | 100 | 265 | 19.4202701740 |
| low-width candidate | 267/267 | 273 | 29 | 302 | 19.2895729826 |

The model is logically exact but not score-efficient. It saves 71 parameters,
while the materialized `1x1` Conv output has `1x1x9x3` float32 elements and
adds 108 bytes to official activation memory. Net cost therefore rises by 37.

An empirical linear-support check over all 801 public input tiles also rules out
the obvious direct-Conv reduction. For a single zero-tested checksum, every
fixed rectangular support from `1x1` through `3x2` has at least one asymmetric
tile in the span of the symmetric examples; only the full `3x3` support has
zero inseparable asymmetric tiles. Therefore a shorter dense kernel cannot
replace the baseline checksum exactly on the public set.

## Decision

Keep the official cost-265 baseline. The low-width candidate is retained as an
exact negative result in `onnx/task146_candidate.onnx`, but it is not accepted.
No generic optimizer or full-grid Gather route was used in round 2.
