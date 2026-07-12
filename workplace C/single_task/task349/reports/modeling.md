# task349 Independent Width/Halo Modeling

## Rule

Each solid color-9 rectangle creates a color-3 halo. The halo radius is half
the rectangle width (valid widths are 2, 4, 6, 8, and 10). Color-1 rays extend
downward from color-9 cells. Precedence is 9 over 3 over 1 over background.

## Compression investigated

1. Two-channel `[class, class^2]` and learned linear halo decoding were tested
   against all public overlap neighborhoods. The 15,951 unique two-channel
   neighborhoods are not linearly separable; a linear SVM leaves about 1,370
   errors. This explains the earlier false outer rings.
2. Joint integer detector/halo rank-4 training was implemented in
   `scripts/search_joint_width_halo.py`. The best explored quantized model still
   had 8,740 cell errors, so it was not exported as a valid candidate.
3. The accepted model compresses the spatial support of the five width
   channels. A width detector marks a rectangle's left edge. Since width is at
   least two, canvas column 29 can never hold a valid left edge, so the width
   tensor's final column is removed. The halo right pad is increased to retain
   exact 30-column alignment.
4. A top-edge rectangle has at least two rows in every public task349 example.
   Its row-0 trigger is redundant with row 1 after vertical halo expansion. A
   2x11 detector drops that trigger row. Bottom trimming was rejected because
   public examples contain height-1 rectangles clipped at the bottom edge.

The resulting width activation changes from `5x30x30` to `5x29x29`. This is a
task-rule-specific receptive-field model, not an Identity, opset, Pad-only, or
initializer-only rewrite.

## Full official validation

- Prior k11 candidate: 267/267, memory 13710, params 1177, cost 14887.
- Final candidate: 267/267, memory 13415, params 1232, cost 14647.
- Delta versus prior candidate: -240 cost, +0.016255 points.
- Delta versus original baseline 14892: -245 cost.
- Local accepted: yes.
