# Cost Diff And Acceptance Protocol

Generated: 2026-07-09T15:58:38

Accepted requires all of:

1. Full local example validation passes.
2. `new_cost < old_cost` under official `neurogolf_utils.py` scoring.
3. Artifact path exists locally.
4. Experiment row is written to `30_SCORE_EXPERIMENT_LEDGER.csv`.

Exploratory rows with missing cost or failed validation are never marked accepted.
