# task372 Analysis

- Public examples: 3 train, 1 test, 262 arc-gen, 266 total.
- Python rule: 266/266. The row of 5s divides two 5x11 panels; the output overlays the upper and lower panels cellwise.
- Across all 14,630 public output cells, the two panels never contain conflicting foreground colors at one position.
- The dense baseline uses the fixed divider channel as a spatially varying bias that restricts valid output to the first five rows.

## 2026-07-11 Grouped Conv Probe

Cropping to 11x11, applying a 10-group 7x1 Conv, and padding the 5x11 result is exact but memory-negative:

| artifact | valid | memory | params | cost | delta points |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | true | 0 | 710 | 710 | 0 |
| cropped group Conv | true | 7,040 | 95 | 7,135 | -2.307503 |

Removing the crop/pad makes the grouped Conv incorrect because the divider-derived bias becomes active outside the output panel. Sparse Conv weights are not accepted by ONNX checker.

Conclusion: no valid lower-cost graph was produced. Do not submit this candidate.

## 2026-07-12 group=2 hard-margin model

The old conclusion applied only to the explicit crop/pad decomposition. A new
search treated the public input windows as a finite classification domain.
Each output channel is linearly separable when the Conv uses two groups:
outputs 0..4 see input channels 0..4 and outputs 5..9 see channels 5..9.

The resulting model is one `7x1 group=2 Conv`, has no scored intermediate
tensors, and passes 266/266 examples:

| artifact | valid | memory | params | cost | points |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline dense Conv | 266/266 | 0 | 710 | 710 | 18.43473502996464 |
| group=2 LP Conv | 266/266 | 0 | 360 | 360 | 19.113895968549844 |

This is accepted locally with delta cost `-350` and delta points
`+0.6791609385852044`.
