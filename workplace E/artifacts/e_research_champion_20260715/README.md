# Research-only champion materialization

This directory records a full-400 research composition generated from the
reproducible `7379.41` parent plus locally lower-cost E candidates.

Do not submit this composition to Kaggle. It contains candidates that were
later rejected by differential fuzzing or hidden-set evaluation, including:

- `task012`: `69/500` color-permutation mismatches.
- `task050`: isolated Kaggle regression `7385.93 -> 7385.89`.
- `task233`: `12/147` comparable color-permutation mismatches.
- `task013`: no visible isolated online gain.

The tracked `manifest.json` preserves every override source, cost, hash, and
evidence label. The generated `onnx/` directory contains 400 models totaling
1,318,622 bytes, but is intentionally not tracked because it duplicates source
models and could be mistaken for a submission-safe package.

Use the accepted model records listed in
`../../TEAM_HANDOFF_20260715.md` for real submissions.
