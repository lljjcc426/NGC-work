# task081 independent modeling

## Rule

Every color-8 connected component is an L triomino occupying three cells of a 2x2 square. The missing fourth corner is filled with color 1 while the original color-8 cells remain unchanged.

## Candidate structure

`task081_float_conv.onnx` implements two oriented missing-corner detectors and the original cyan identity feature as a float Conv network. A clipped hidden layer is centered and projected by a second Conv into output channels 0, 1, and 8, then padded to the canonical 30x30 tensor. This replaces both quantized convolutions with a separate float rule network.

## Validation

- Public examples: 2 train, 1 test, 261 arc-gen.
- Candidate: 264/264 exact.
- Baseline cost: 464 (memory 392, params 72).
- Candidate cost: 6452 (memory 6370, params 82).
- Decision: rule accepted as an independent model; replacement rejected because float feature maps are much more expensive than the quantized baseline.

## Round 2: quantized hard-margin compression

The 264 public examples reduce to 93 unique 3x3 cyan neighborhoods with no
identical-window label conflicts. Exact LP hard-margin tests give:

| kernel | color 0 | color 1 | color 8 | direct weight parameters |
| --- | --- | --- | --- | ---: |
| 3x3 | infeasible | infeasible | feasible | 90 |
| 5x5 | infeasible | infeasible | feasible | 250 |
| 7x7 | feasible | feasible | feasible | 490 |

The 3x3 and 5x5 failures are not optimizer failures: minimized infeasible cores
are recorded in `hard_margin_counterexamples.csv` (four windows per output class
at 3x3; six per output class at 5x5). The 7x7 case cannot reduce official cost,
because its 490 weights alone exceed the complete baseline cost of 464.

A two-hidden-channel float ReLU model can classify all 93 unique neighborhoods.
However, its integerized hidden codes still need each one-vs-rest final score to
fit a single uint8 output bucket. A deterministic search over 79 scales and 300
integer perturbations per scale checked 23,700 candidates and found no candidate
where colors 0, 1, and 8 all satisfied those bounded hard-margin constraints.

No lower-cost task081 replacement was produced. The three-channel quantized
baseline remains the best validated artifact.
