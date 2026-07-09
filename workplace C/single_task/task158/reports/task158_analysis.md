# task158 Analysis

- task_json: `E:\kongming\NGC-work\neurogolf_400_tasks\tasks\task158.json`
- examples: train `3`, test `1`, arc-gen `262`
- output colors subset of input colors: `True`
- changed cells min/median/max: `3` / `15` / `90`
- changed ratio avg: `0.041779`
- examples with a 3+ color connected component: `87/266` (0.327)

## Interpretation

- Background is the dominant color per example.
- Outputs are same-shape and preserve most input cells.
- The recurring structure is a 3x3 motif/template containing two corner marker colors and one bridge/fill color.
- Other same-color rectangular marker components appear elsewhere; the output overlays a rotated/reflected/scaled copy of the motif between matching marker components.

## Changed Color Edges

| before | after | count |
| ---: | ---: | ---: |
| 2 | 4 | 242 |
| 3 | 4 | 189 |
| 4 | 9 | 179 |
| 4 | 6 | 166 |
| 1 | 8 | 151 |
| 5 | 8 | 147 |
| 5 | 7 | 130 |
| 3 | 9 | 126 |
| 9 | 5 | 125 |
| 5 | 3 | 125 |
| 2 | 6 | 119 |
| 4 | 2 | 119 |
| 4 | 1 | 113 |
| 6 | 9 | 110 |
| 7 | 9 | 107 |
| 3 | 8 | 104 |
| 8 | 4 | 104 |
| 5 | 1 | 95 |
| 2 | 8 | 95 |
| 6 | 3 | 94 |
| 1 | 5 | 92 |
| 2 | 9 | 89 |
| 9 | 1 | 85 |
| 1 | 2 | 85 |
| 6 | 8 | 83 |
| 9 | 6 | 80 |
| 3 | 5 | 80 |
| 3 | 1 | 79 |
| 5 | 6 | 71 |
| 6 | 4 | 71 |
| 9 | 4 | 70 |
| 1 | 4 | 67 |
| 9 | 2 | 66 |
| 7 | 5 | 66 |
| 6 | 2 | 64 |
| 7 | 2 | 57 |
| 3 | 7 | 55 |
| 9 | 3 | 53 |
| 6 | 5 | 52 |
| 4 | 7 | 51 |
| 5 | 2 | 51 |
| 6 | 7 | 50 |
| 8 | 6 | 48 |
| 3 | 2 | 48 |
| 1 | 7 | 47 |
| 8 | 2 | 47 |
| 5 | 4 | 46 |
| 9 | 7 | 45 |
| 7 | 6 | 43 |
| 8 | 1 | 40 |
| 7 | 4 | 39 |
| 4 | 8 | 39 |
| 4 | 5 | 34 |
| 7 | 3 | 33 |
| 8 | 7 | 33 |
| 1 | 3 | 32 |
| 8 | 3 | 31 |
| 3 | 6 | 29 |
| 4 | 3 | 28 |
| 2 | 3 | 27 |
| 7 | 8 | 22 |
| 1 | 6 | 20 |
| 2 | 5 | 19 |
| 2 | 7 | 18 |
| 6 | 1 | 17 |
| 8 | 5 | 15 |
| 7 | 1 | 11 |
| 8 | 9 | 5 |
| 1 | 9 | 4 |
| 2 | 1 | 4 |
| 5 | 9 | 3 |
| 9 | 8 | 3 |

## Component Signature Top 20

| count | colors | shape | mask |
| ---: | --- | --- | --- |
| 82 | `{2: 1}` | `(1, 1)` | `2` |
| 81 | `{5: 1}` | `(1, 1)` | `5` |
| 69 | `{6: 1}` | `(1, 1)` | `6` |
| 68 | `{7: 1}` | `(1, 1)` | `7` |
| 65 | `{1: 1}` | `(1, 1)` | `1` |
| 63 | `{4: 1}` | `(1, 1)` | `4` |
| 63 | `{8: 1}` | `(1, 1)` | `8` |
| 58 | `{3: 1}` | `(1, 1)` | `3` |
| 55 | `{9: 1}` | `(1, 1)` | `9` |
| 34 | `{5: 4}` | `(2, 2)` | `55;55` |
| 30 | `{4: 4}` | `(2, 2)` | `44;44` |
| 29 | `{3: 4}` | `(2, 2)` | `33;33` |
| 28 | `{2: 4}` | `(2, 2)` | `22;22` |
| 26 | `{1: 4}` | `(2, 2)` | `11;11` |
| 26 | `{6: 4}` | `(2, 2)` | `66;66` |
| 23 | `{7: 4}` | `(2, 2)` | `77;77` |
| 20 | `{9: 4}` | `(2, 2)` | `99;99` |
| 18 | `{7: 9}` | `(3, 3)` | `777;777;777` |
| 18 | `{8: 4}` | `(2, 2)` | `88;88` |
| 15 | `{6: 9}` | `(3, 3)` | `666;666;666` |

## Neighborhood Features

- 3x3 and 5x5 counters are computed and available for follow-up in this script; the markdown keeps only aggregate interpretations to stay readable.
- High-frequency changed cells are background cells adjacent to existing marker or motif blocks, consistent with mask overlay rather than global recoloring.
