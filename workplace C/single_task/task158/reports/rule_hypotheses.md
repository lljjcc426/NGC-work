# task158 Rule Hypotheses

## h1_sparse_mask_overlay

- Description: Strict 3x3 motif source, opposite-corner endpoint colors, square marker pair overlay.
- Passed: 266 / 266
- Split pass counts: train 3/3, test 1/1, arc-gen 262/262

## h2_connected_component_bbox

- Description: Connected-component source bbox rule; expected to fail when motif contains background holes.
- Passed: 116 / 266
- Split pass counts: train 2/3, test 0/1, arc-gen 114/262

## h3_marker_driven_motif_copy

- Description: Input-only source window selection plus D4 transform and square marker-pair scaled overlay.
- Passed: 266 / 266
- Split pass counts: train 3/3, test 1/1, arc-gen 262/262

