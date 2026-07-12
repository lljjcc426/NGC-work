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
