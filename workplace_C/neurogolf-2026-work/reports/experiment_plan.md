# Experiment Plan

Status: provisional until the official competition metric and data structure are fetched.

Do not treat the model choices below as final; they must be updated after `reports/data_profile.md` and `reports/neurogolf-2026_competition_brief.md` contain real Kaggle-derived facts.

1. `baseline_auto`: run `python src/make_baseline.py` and validate the generated submission. Use only as a pipeline and format check until the official metric is known.
2. `stronger_features`: derive task-specific features from the true train/test schema. For tabular data, start with missingness flags, grouped statistics, and leakage checks. For non-tabular data, replace this with format-specific preprocessing.
3. `stronger_model`: choose LightGBM/XGBoost/CatBoost or a neural model only after the data modality and competition rules are known.
4. `cv_strategy`: match the official metric and leaderboard split as closely as the data supports. Use grouped/time-aware folds only if the real schema shows group or time fields.
5. `ensemble_blend`: blend multiple seeds, model families, or feature sets after at least two independently validated experiments beat baseline.
6. `post_processing`: add only if the rules, target constraints, and sample submission support it.
7. `ablation`: remove each feature/model/post-processing change to confirm it improves CV and does not only fit public LB noise.

Experiment records are stored in `reports/experiments.csv`.
