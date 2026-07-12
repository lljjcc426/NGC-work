# FINAL task081 report

## Result

- Rule: complete each cyan L triomino's missing 2x2 corner with color 1.
- Official baseline validation: 264/264 exact, cost 464 (memory 392, params 72).
- Independent float rule validation: 264/264 exact, cost 6452.
- Lower-cost replacement accepted: no.

## Quantized hard-margin search

- The public corpus contains 93 unique 3x3 cyan neighborhoods.
- A direct one-stage 3x3 or 5x5 affine threshold is infeasible for output colors 0 and 1. Reproducible minimal infeasible subsets are saved in `hard_margin_counterexamples.csv`.
- A 7x7 direct threshold is sign-separable, but a ten-output convolution requires at least `10 * 7 * 7 = 490` weight parameters. This already exceeds the complete baseline cost of 464 before bias, constants, or activation memory, so it cannot improve cost.
- A width-2 float ReLU classifier can classify the 93 neighborhoods, but quantized output requires each class's positive scores to fit one uint8 bucket. A seeded search over 79 scales and 300 integer perturbations per scale checked 23,700 two-channel kernels; none made colors 0, 1, and 8 simultaneously separable into valid bounded output buckets.
- Cropping is invalid: valid cyan and generated color-1 outputs occur on all four edges of the 7x7 active area.

## Reproduction

```powershell
python scripts/search_quantized_hard_margin.py
python scripts/search_quantized_hard_margin.py --two-channel-trials 300
```

Evidence:

- `reports/hard_margin_results.csv`
- `reports/hard_margin_counterexamples.csv`
- `reports/hard_margin_search_summary.json`
- `reports/cost_diff_round2.csv`

The current two-stage three-channel quantized baseline remains the lowest validated task081 model.
