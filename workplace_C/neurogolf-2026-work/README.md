# neurogolf-2026 Work

Engineering workspace for Kaggle competition `neurogolf-2026`, located under repository `workplace_C`.

Current status: initialized, NVIDIA `nvidia-kaggle` reference checkout available, Kaggle access blocked because `KAGGLE_API_TOKEN` is not set.

See `reports/stage_status.md` for the exact completed stages and blockers.

## Directory Structure

```text
neurogolf-2026-work/
├── README.md
├── reports/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── src/
├── submissions/
├── logs/
├── scripts/
├── external/nvidia-kaggle/
└── kaggle_kernel/
```

## Environment

```powershell
python --version
pip --version
python -m pip install pandas numpy scikit-learn matplotlib tqdm pyarrow kaggle httpx kagglesdk nbformat pydantic python-dotenv rich
```

Do not print or commit `KAGGLE_API_TOKEN`.

```powershell
python -c "import os; assert os.environ.get('KAGGLE_API_TOKEN'), 'KAGGLE_API_TOKEN is not set'; print('KAGGLE_API_TOKEN is set')"
```

## NVIDIA Kaggle Skill

The Codex marketplace was added with:

```powershell
codex plugin marketplace add https://github.com/NVIDIA/nvidia-kaggle.git
```

Because the active session did not expose the plugin tools directly, the official repository is checked out locally:

```powershell
git clone git@github.com:NVIDIA/nvidia-kaggle.git external/nvidia-kaggle
```

## Data Download

After setting `KAGGLE_API_TOKEN` and accepting the competition rules on Kaggle if required:

```powershell
kaggle competitions files -c neurogolf-2026 | Tee-Object -FilePath logs\kaggle_files.txt
kaggle competitions download -c neurogolf-2026 -p data\raw
python -c "from pathlib import Path; import zipfile; raw=Path('data/raw'); [zipfile.ZipFile(z).extractall(raw/z.stem) for z in raw.glob('*.zip')]"
Get-ChildItem data\raw -Recurse -File | Sort-Object FullName | ForEach-Object { $_.FullName } | Tee-Object -FilePath reports\data_file_manifest.txt
```

Or run the bounded resume pipeline:

```powershell
.\scripts\run_after_kaggle_token.ps1
```

This script stops after submission quota check and does not submit.

## Competition Research

Run from project root:

```powershell
Push-Location external\nvidia-kaggle\skills\nvidia-kaggle-skill
python .\scripts\fetch_competition_info.py neurogolf-2026 > ..\..\..\..\reports\neurogolf-2026_competition_overview_raw.txt
python .\scripts\fetch_dataset_info.py neurogolf-2026 > ..\..\..\..\reports\neurogolf-2026_dataset_description_raw.txt
Pop-Location
```

Then update `reports/neurogolf-2026_competition_brief.md` with only facts confirmed by those outputs.

## Data Check

```powershell
python src\data_check.py
```

Outputs:

- `reports/data_profile.md`
- `reports/data_profile.json`

## Baseline

```powershell
python src\make_baseline.py 2>&1 | Tee-Object -FilePath logs\baseline_run.log
python src\validate_submission.py submissions\submission_baseline.csv
```

The baseline script auto-detects train/test/sample submission files. If no target can be confirmed, it only creates a format-check baseline and marks the report accordingly.

## Kaggle Kernel Build

Only use this after confirming the competition is Notebook-only or Code Competition:

```powershell
python scripts\build_kaggle_kernel.py --output-file submission.csv
```

Outputs:

- `kaggle_kernel/kernel-metadata.json`
- `kaggle_kernel/neurogolf_2026_baseline.py`

## Submission Rules

Always validate locally first:

```powershell
python src\validate_submission.py submissions\submission_baseline.csv
```

Check quota before any submit:

```powershell
Push-Location external\nvidia-kaggle\skills\nvidia-kaggle-skill
python .\scripts\submission_quota.py neurogolf-2026 --by-user --by-day --as-json > ..\..\..\..\reports\submission_quota.json
Pop-Location
```

No submission should be run until the user explicitly says: `提交`.

## Current Known Score

UNKNOWN. No Kaggle submission has been made and no public LB score has been fetched.

## Next Experiments

See `reports/experiment_plan.md`. The plan is provisional until the official metric and data schema are confirmed.
