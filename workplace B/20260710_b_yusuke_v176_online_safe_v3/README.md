# B Online-Safe V3

Date: 2026-07-10

Scope: B tasks only.

Working baseline:

- Local source: `yusuketogashi_v176_full`
- Kaggle public baseline: `7267.31`
- Local scorer baseline: `7267.145162`

Submitted candidate:

- File: `submission.zip`
- Kaggle ref: `54517873`
- Online public score: `7267.85`
- Online gain vs yusuke public baseline: `+0.54`
- Local gain vs baseline: `+0.569119`
- Expected local total: `7267.714281`
- SHA256: `febafe5f6a6cf84640b35ce811c5cd6589a4e2fedfe4d15f9b9aac95176a1273`

Changed B Tasks

| task | method | cost | local gain | online probe |
| --- | --- | ---: | ---: | --- |
| `task205` | `float16` final-tail rewrite | `4891 -> 4251` | `+0.140243` | with task377: ref `54517745`, score `7267.56` |
| `task255` | `uint8/QLinearMatMul` tail rewrite | `8911 -> 7532` | `+0.168126` | ref `54517673`, score `7267.45` |
| `task277` | prune one A/B MaxPool iteration | `3540 -> 3140` | `+0.119904` | ref `54517695`, score `7267.40` |
| `task377` | `float16` final-tail rewrite | `4567 -> 3967` | `+0.140846` | with task205: ref `54517745`, score `7267.56` |

Why V3 Is The Safe Set

- `task255`, `task277`, and `task205+task377` were each online-positive as isolated probes.
- The combined four-task package scored `7267.85`, so the changes survived interaction online.
- `task023` is excluded even though it is locally positive: the no-uint8-TopK debug package scored only `7251.58`, matching the hidden-risk pattern.
- `task076` and `task285` are excluded because the `uint8 TopK` rewrites caused Kaggle `ERROR` in the larger bundles.

Included Evidence

- `summary.json`: local score deltas, source paths, zip hash, and excluded-task notes.
- `changed_tasks.csv`: compact per-task scoring table.
- `overrides/`: the four ONNX files inserted into the full package.
- `scripts/`: rewrite scripts for the accepted four tasks.
- `reports/`: scorer artifacts for accepted rewrites and one rejected `task018` scan.
- `online_probes/`: summary tables for the single/paired online probes and the two failure-debug packages.

Next Direction

Keep working B-only and self-rewrite first. The next real targets are:

1. `task018`: largest B headroom, but old label-tail/TopK ideas do not match V176 safely.
2. `task101`: current old ARC-GEN generator is worse than V176; needs graph-local simplification, not full replacement.
3. `task350`: compact graph with high memory in final full output; look for boolean/one-hot tail reduction.
4. `task209` and `task328`: inspect current graph names directly because older transform modules do not hit V176.

If the team can recover the current global package behind ref `54517518` (`7268.99`), these four B overrides should be tested on top of that base instead of yusuke V176.
