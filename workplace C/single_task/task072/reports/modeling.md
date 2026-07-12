# task072 independent modeling

## Rule

Rows 0-5 and rows 7-12 are two 6x5 binary pictures separated by a color-4 row. The output is their cellwise XOR: differing cells become color 3 and equal cells become color 0.

## Candidate structure

`task072_xor.onnx` reads only input channel 2 from the two pictures, compares the two boolean planes, negates equality to obtain XOR, and directly constructs output channels 0 and 3. This discards the baseline's four-channel slices and per-channel Where constants.

## Validation

- Public examples: 4 train, 2 test, 262 arc-gen.
- Candidate: 268/268 exact.
- Baseline cost: 421 (memory 390, params 31).
- Candidate cost: 474 (memory 450, params 24).
- Decision: rule accepted as an independent model; replacement rejected because cost increased by 53 despite fewer parameters.
