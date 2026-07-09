# task158 Rule Validation

Solver: `h3_marker_driven_motif_copy` / `solve()`

| split | passed | total |
| --- | ---: | ---: |
| train | 3 | 3 |
| test | 1 | 1 |
| arc-gen | 262 | 262 |
| all | 266 | 266 |

The rule is input-only: detect the dominant background, find a 3x3 motif with two opposite-corner endpoint colors and one fill color, then use square endpoint-color marker components to place scaled D4-transformed sparse overlays.
