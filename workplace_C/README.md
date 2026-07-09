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
- `neurogolf-2026-work/`: Kaggle competition engineering workspace and reproducibility scaffold.
- `cleanup_log.md`: cleanup record for setup-only directories removed from the repository root.

## Current Blocker

`KAGGLE_API_TOKEN` is not set in the current shell. Kaggle API/CLI operations, competition metadata fetches, data downloads, public notebook research, quota checks, and submissions are intentionally blocked until the token is configured.

No Kaggle submission has been made.

## Resume

After setting `KAGGLE_API_TOKEN`, continue from:

```powershell
cd workplace_C\neurogolf-2026-work
.\scripts\run_after_kaggle_token.ps1
```

The script stops at quota check. It does not submit.
