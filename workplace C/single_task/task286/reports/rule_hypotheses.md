# task286 Rule Hypotheses

## h1_seeded_checker_fill

- Description: Multi-source BFS from marker cells through non-wall cells; alternate two marker colors by graph distance.
- Passed: 265 / 265
- Split pass counts: train 2/2, test 1/1, arc-gen 262/262

## h2_straight_line_extension

- Description: Only extend colors along straight corridors from each marker; fails at turns/branches.
- Passed: 6 / 265
- Split pass counts: train 0/2, test 0/1, arc-gen 6/262

## h3_component_plain_fill

- Description: Fill the marker-connected component with one marker color; fails because parity matters.
- Passed: 1 / 265
- Split pass counts: train 0/2, test 0/1, arc-gen 1/262

