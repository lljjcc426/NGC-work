# task286 Rule Validation

Solver: `h1_seeded_checker_fill` / `solve()`

| split | passed | total |
| --- | ---: | ---: |
| train | 2 | 2 |
| test | 1 | 1 |
| arc-gen | 262 | 262 |
| all | 265 | 265 |

The rule is input-only: treat `8` as wall, `0` as corridor, detect the two marker colors, and fill only marker-connected corridors with alternating marker colors by 4-neighbor graph distance.
