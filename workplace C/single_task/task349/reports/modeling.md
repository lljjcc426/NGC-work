# task349 Collision-Safe Width/Halo Model

## Rule structure

Solid color-9 rectangles have widths 2, 4, 6, 8, or 10. Each rectangle emits
a color-3 halo with radius `width / 2`, while color-1 rays extend downward.
Color precedence is `9 > 3 > 1 > 0`.

The prior valid graph represented the five widths as five one-hot activation
channels. After spatial trimming it cost 14647, of which the `5x29x29` width
activation alone used 4205 bytes.

## Explicit collision constraints

All 267 examples produce 87 unique horizontal detector neighborhoods. Integer
constraints were imposed on every neighborhood, including shifted and
overlapping rectangles:

- width 2 and width 6 emit code `1`;
- width 4 and width 8 emit code `127` in their respective merged channels;
- all unrelated or shifted patterns have a non-positive detector accumulator;
- halo bias is `-124`;
- common inner support uses weight `127`, giving minimum true value `3`;
- the larger-only support uses weight `1`, so as many as 124 smaller-width
  collision contributions remain non-positive.

This safely merges widths `2/4` and `6/8`; width 10 remains a third channel.
One shared halo convolution consumes all three channels. The graph still has no
new full-grid branch or mask.

## Results

| model | width activation | memory | params | cost | validation |
| --- | --- | ---: | ---: | ---: | ---: |
| prior spatial candidate | `5x29x29` | 13415 | 1232 | 14647 | 267/267 |
| single merged pair | `4x29x29` | 12574 | 990 | 13564 | 267/267 |
| two merged pairs | `3x29x29` | 11733 | 747 | 12480 | 267/267 |

The accepted model lowers cost by 2167 and adds approximately 0.160108 points
relative to the 14647 starting point.

## Rejected two-channel branch

Widths `6/8/10` were also encoded as amplitudes `1/2/127`, reducing local cost
to 11396. It passed only 192/267 because two width-6 outer-ring contributions
can cross the threshold intended for width 8. This artifact is retained only as
a documented collision counterexample and is not accepted.
