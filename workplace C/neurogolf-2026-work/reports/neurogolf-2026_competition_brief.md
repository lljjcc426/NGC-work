# neurogolf-2026 Competition Brief

Status: BLOCKED before Kaggle API/CLI fetch because `KAGGLE_API_TOKEN` is not set.

No competition rules, evaluation metric, timeline, data fields, or submission format have been inferred.

## Required Facts

1. Competition objective: UNKNOWN
2. Prediction target: UNKNOWN
3. Evaluation metric: UNKNOWN
4. Score direction: UNKNOWN
5. Submission file name and fields: UNKNOWN
6. Sample submission format: UNKNOWN
7. Data file list: UNKNOWN
8. Data size: UNKNOWN
9. Code Competition / Notebook-only: UNKNOWN
10. Internet allowed: UNKNOWN
11. GPU required: UNKNOWN
12. Daily submission limit: UNKNOWN
13. Final deadline: UNKNOWN
14. Special rules or prohibited actions: UNKNOWN

## Commands Attempted

- `codex plugin marketplace add https://github.com/NVIDIA/nvidia-kaggle.git`
  - Result: marketplace added successfully.
- `git clone git@github.com:NVIDIA/nvidia-kaggle.git external/nvidia-kaggle`
  - Result: local reference checkout created at `external/nvidia-kaggle`, HEAD `d91e899`.
- `python -c "... KAGGLE_API_TOKEN ..."`
  - Result: `KAGGLE_API_TOKEN is not set`.

## Next Command After Token Is Set

Run from project root:

```powershell
python -c "import os; assert os.environ.get('KAGGLE_API_TOKEN'), 'KAGGLE_API_TOKEN is not set'; print('KAGGLE_API_TOKEN is set')"
```

Then run the NVIDIA skill fetch scripts from `external/nvidia-kaggle/skills/nvidia-kaggle-skill`.
