# Workplace D Next Opportunities

Updated: 2026-07-09 13:20 +0800

## Summary

The first D scan found only one directly adoptable replacement from the local candidate pool:

- `task029`: cost `5288 -> 5284`, points `+0.000756334`.

This does **not** mean D has no remaining optimization potential. It means no second candidate in the available local pool beat the assignment baseline from `v244_costs.txt`.

Important caveat: the local NeuroGolf workspace mainly contains artifacts up to v157/v158, while the assignment baseline was generated from v244 costs. Several D tasks already have lower v244 costs than any locally available ONNX candidate, so direct replacement scanning cannot improve them without either:

- access to the actual v244/v245+ submission artifacts, or
- new task-specific rule mining / ONNX graph rewrites.

## Highest-Value D Targets

These are the best remaining D tasks to attack manually, ordered by assignment cost / gap.

| task | baseline cost | points | gap_to_18 | current observation | recommended next move |
| --- | ---: | ---: | ---: | --- | --- |
| `task133` | 21278 | 15.034571 | 2.965429 | Local v157 candidate is much worse than v244 baseline; current graph appears to locate bounding boxes and code regions. | Need v244 artifact or re-mine the rule from examples; not a simple source graft. |
| `task002` | 14285 | 15.433035 | 2.566965 | Local best candidate is `14605`, still worse than v244. Rule adds color `4` into specific gaps among color `3` structures. | Manual rule graph may be possible, but current bitwise graph is already compact. |
| `task243` | 13654 | 15.478212 | 2.521788 | Local candidate pool is far worse; existing graphs are large bitwise encodings. | Requires dedicated rule mining or latest artifact. |
| `task173` | 13383 | 15.498259 | 2.501741 | Local candidates much worse; graph uses Gather/Scatter/Where style edits. | Inspect ARC examples and search for simpler geometric condition. |
| `task145` | 12855 | 15.538512 | 2.461488 | Output introduces colors `1` and `8`; current local graph uses MaxPool/Where. | Good manual candidate: same-shape binary-color task with added colors. |
| `task074` | 9050 | 15.889480 | 2.110520 | Very small 8-node graph, but high memory because it materializes color-index symmetries. | Hard to beat with standard ops; possible only with lower-memory symmetry construction. |
| `task219` | 8739 | 15.924449 | 2.075551 | Adds color `1` to an `8`/background pattern; local graph has many reductions/gathers. | Good manual candidate if rule can be expressed with local neighborhoods. |

## Near-Saturated Tasks

Many D tasks are already at the assignment baseline in local candidates and are unlikely to pay off without very targeted micro-optimizations:

- Zero-cost delta candidates found for `task028`, `task032`, `task045`, `task047`, `task073`, `task082`, `task087`, `task106`, `task114`, `task115`, `task127`, `task147`, `task155`, `task160`, `task164`, `task166`, `task167`, `task229`, `task232`, `task256`, `task282`, `task292`, `task314`, `task331`, `task386`, `task400`.

## Practical Next Steps

1. Obtain the actual v244/v245+ submission artifact used to create `assignments/task_assignment_400.csv`.
2. Re-run `d_optimize_rule_mining.py` with that artifact as an additional source root and as the true base for equivalence checks.
3. Prioritize manual rule mining for `task145`, `task219`, and `task002`; these have visible rule structure and meaningful remaining gap.
4. Treat `task074` as a graph-compression challenge rather than a rule-discovery task; the rule is compact but memory-heavy.

