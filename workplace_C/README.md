# Workplace C

Owner track: `onnx_equiv_compression`

This workspace contains the C-group work package for the 2026 NeuroGolf Championship assignments listed in the repository root README.

## Scope

- Primary source of task ownership: `../assignments/task_assignment_400.csv`
- Assigned owner: `C`
- Assignment type: `primary`
- Task count: 67
- Working project: `neurogolf-2026-work/`

## Artifacts

- `task_manifest_C.csv`: C task rows extracted from the assignment CSV.
- `task_manifest_C.md`: C task summary grouped by priority and structure metadata.
- `dashboard/`: read-only C task progress dashboard exported from the local NeuroGolf task table.
- `reports/onnx_visualization_sources.md`: Kaggle discussion evidence for ONNX visualization / GUI tooling.
- `neurogolf-2026-work/`: Kaggle competition engineering workspace and reproducibility scaffold.
- `cleanup_log.md`: cleanup record for setup-only directories removed from the repository root.

## Dashboard

Open `dashboard/index.html` directly, or serve it locally:

```powershell
cd workplace_C\dashboard
.\serve_dashboard.ps1
```

The browser page refreshes every 30 seconds. Regenerate `task_progress_C.csv` and `index.html` with:

```powershell
cd workplace_C\dashboard
.\refresh_from_kagglegolf.ps1
```

## Current Kaggle API Blocker

The standalone `neurogolf-2026-work/` scaffold still expects `KAGGLE_API_TOKEN` before running Kaggle API/CLI operations, competition metadata fetches, data downloads, public notebook research, quota checks, or submissions.

The `dashboard/` artifacts are read-only exports and do not require Kaggle credentials.

## Resume

After setting `KAGGLE_API_TOKEN`, continue from:

```powershell
cd workplace_C\neurogolf-2026-work
.\scripts\run_after_kaggle_token.ps1
```

The script stops at quota check. It does not submit.
