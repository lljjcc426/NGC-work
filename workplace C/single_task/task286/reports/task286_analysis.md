# task286 Analysis

- task_json: `E:\kongming\NGC-work\neurogolf_400_tasks\tasks\task286.json`
- examples: train `2`, test `1`, arc-gen `262`
- baseline current_cost: `26909`
- baseline current_points: `14.799783917876258`
- output colors subset of input colors: `True`
- changed ratio avg: `0.353185`
- changed cells min/median/max: `1` / `100` / `301`

## Interpretation

- `8` behaves as wall/barrier color.
- `0` behaves as unfilled corridor color.
- The two non `{0,8}` colors are seed/marker colors.
- Output fills only the passable connected component containing marker cells.
- Fill color alternates by 4-neighbor graph distance from the seed markers.
- Other passable components without markers remain `0`.

## Marker Color Pairs

| marker_colors | examples |
| --- | ---: |
| `3 7` | 16 |
| `4 5` | 15 |
| `3 5` | 15 |
| `3 4` | 14 |
| `1 3` | 14 |
| `7 9` | 13 |
| `2 3` | 12 |
| `1 5` | 12 |
| `5 9` | 12 |
| `4 7` | 11 |
| `1 7` | 11 |
| `1 6` | 11 |
| `6 7` | 10 |
| `1 9` | 10 |
| `1 2` | 9 |
| `2 9` | 9 |
| `4 9` | 9 |
| `1 4` | 7 |
| `6 9` | 7 |
| `2 6` | 7 |
| `2 4` | 7 |
| `2 7` | 6 |
| `5 7` | 6 |
| `3 6` | 6 |
| `2 5` | 6 |
| `4 6` | 4 |
| `5 6` | 4 |
| `3 9` | 2 |

## Component Counts

| passable_components | seeded_components | examples |
| ---: | ---: | ---: |
| 5 | 1 | 35 |
| 2 | 1 | 35 |
| 3 | 1 | 35 |
| 4 | 1 | 31 |
| 6 | 1 | 21 |
| 7 | 1 | 20 |
| 8 | 1 | 20 |
| 1 | 1 | 17 |
| 9 | 1 | 16 |
| 13 | 1 | 11 |
| 11 | 1 | 11 |
| 10 | 1 | 8 |
| 12 | 1 | 2 |
| 16 | 1 | 1 |
| 14 | 1 | 1 |
| 15 | 1 | 1 |
