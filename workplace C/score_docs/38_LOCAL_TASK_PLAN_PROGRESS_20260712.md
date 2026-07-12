# C Local Task Plan Progress - 2026-07-12

This pass is local-only. No parent package, kernel, quota query, queue, or
Kaggle submission was used.

## Accepted

| task | dedicated rewrite | old cost | new cost | delta points | validation |
| --- | --- | ---: | ---: | ---: | ---: |
| task332 | fixed crop folded into row-code Conv | 561 | 438 | +0.247501995 | 267/267 |
| task237 | packed Conv emits only active nine rows | 1836 | 1716 | +0.067593291 | 266/266 |
| task091 | spatial-only opset-18 Pad controls | 2759 | 2730 | +0.010566686 | 266/266 |
| task009 | compact separator Pad controls | 6595 | 6585 | +0.001517451 | 265/265 |
| task158 | four orientations encoded in three channels | 28023 | 26250 | +0.065359613 | 266/266 |
| task072 | single 8x1 top/bottom difference Conv | 421 | 368 | +0.134549896 | 268/268 |

This pass adds `+0.5270889313257321` expected local points. The cumulative
unstacked C improvements recorded across the current local campaign are
`+3.965123086040556` points.

## Rejected With Full Evidence

| task | probe | nominal result | validation | conclusion |
| --- | --- | --- | ---: | --- |
| task015 | grouped hard-margin windows | lower parameter target | infeasible | exact duplicate windows require different labels |
| task054 | duplicate-line union | 26756 | full valid | cost increased |
| task054 | line-first neighborhood restore | 24795 | invalid | duplicate-index collision semantics changed |
| task364 | fused seed expansion | 12882 | 110/266 | seed branches carry distinct boundary state |
| task383 | row-only activation crop | 5642 | 257/266 | cropped tail carries sentinel state |
| task383 | column-only activation crop | 5640 | 264/266 | cropped tail carries sentinel state |
| task383 | joint activation crop | 5286 | 242/266 | both boundary failures combine |
| task146 | low-width exact checksum | 302 | 267/267 | 71 params saved but 108 memory added; baseline 265 wins |
| task081 | one-stage quantized hard-margin | infeasible below baseline | exhaustive public neighborhoods | 3x3/5x5 not separable; 7x7 weights alone cost 490 |

## Next Local Targets

1. Revisit task121 with direct nine-cell Gather output and no full patch.
2. Inspect task382 shift banks for a task-specific shared-index contraction.
3. Search task065's scalar bit-field decoder for shared coordinate/color bits.
4. Test task075's placement tensor for an exact low-rank contraction.
5. Preserve task052/task081/task108/task146 baselines unless an algebraic factorization reduces
   initializer count without creating an intermediate spatial tensor.
