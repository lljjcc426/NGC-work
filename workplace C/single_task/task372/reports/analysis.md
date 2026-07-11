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
