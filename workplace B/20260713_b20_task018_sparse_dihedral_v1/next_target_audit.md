# Next-target structural audit

This audit was performed after the task018 rewrite, using the accepted 7296.04
package as the baseline. It records why several low-scoring B tasks were not
immediately rewritten, so the same dead ends are not repeated.

| Task | Current cost | Cost needed for +1 | Structural conclusion |
| --- | ---: | ---: | --- |
| 101 | 13725 | about 5050 | Exact sparse coordinate generator already used. The alternative scaled-sprite convolution model is correct but costs 44802. |
| 076 | 12825 | about 4718 | Current fixed sparse template matcher is substantially smaller than the correct general generator. Existing safe pruning adds only about 0.055. |
| 209 | 7631 | about 2807 | Dynamic corner crop plus resized thumbnail; two GridSample outputs are the main irreducible fields. |
| 023 | 6353 | about 2337 | 64-bit exact-cover solver for overlapping 2x2, 1x3, and 3x1 toys. The 430 nodes are tiny bitboard states, not dense redundant tensors. |
| 131 | 3873 | about 1425 | Move object to a red line and draw an 8 separator. NCHW ScatterND needs 50 scalar channel updates; its int64 index tensor alone costs 1600. |
| 328 | 5746 | about 2114 | Already a direct four-corner Chebyshev Voronoi/parity formula over an 18x18 field. |
| 285 | 19700 | about 7247 | A fresh sparse rewrite is plausible, but a general 30x30 label projection already costs 4500. Do not use the known unsafe no-Pad shortcut. |
| 277 | 3140 | about 1155 | Nine masked flood-fill iterations identify the odd connected component. A row-bitboard rewrite only saves tens of bytes per iteration. |

Best next research direction:

1. Continue task285 only with a whitelist-safe sparse coordinate design that
   preserves the full 30x30 boundary behavior.
2. Accumulate task018 with independently proven terminal-rule rewrites from
   other B tasks until the aggregate gain exceeds +1.0.
3. Do not submit task018 alone; its current verified gain is +0.246033.

