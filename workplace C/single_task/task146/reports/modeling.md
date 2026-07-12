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
