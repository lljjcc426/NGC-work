# Workplace C

Competition archive navigation: [team postmortem](../docs/postmortem/2026-neurogolf-retrospective.md) | [repository guide](../docs/repository-guide.md).

Owner track: `onnx_equiv_compression`

This is the canonical C-group workspace. The earlier duplicate
`workplace_C/` directory has been merged here.

## Scope

- Assignment source: `../assignments/task_assignment_400.csv`
- Assigned owner: `C`
- Assignment type: `primary`
- Task count: 67
- Full task mirror for all teams: `../neurogolf_400_tasks/`

## C Task Files

- `tasks/task009.json` ... `tasks/task392.json`: complete downloaded JSON
  files for the 67 C primary tasks.
- `task_index_C.csv`: C task file checksums, example counts, assignment
  priority, cost, and score metadata.
- `task_viewer_C.html`: local browser viewer for the 67 C tasks.

Start the C task viewer:

```powershell
python -m http.server 8771 --bind 127.0.0.1 --directory "E:\kongming\NGC-work\workplace C"
```

Then open `http://127.0.0.1:8771/task_viewer_C.html`.

## Progress Dashboard

- `dashboard/index.html`: C task progress dashboard exported from the local
  NeuroGolf task table.
- `dashboard/task_progress_C.csv`: machine-readable C progress table.
- `dashboard/task_progress_C.md`: C progress summary.
- `dashboard/source_task_scoreboard_all_tasks.csv`: all-task score table used
  for the C export.

Start the dashboard:

```powershell
cd "E:\kongming\NGC-work\workplace C\dashboard"
.\serve_dashboard.ps1
```

Then open `http://127.0.0.1:8766/index.html`.

Regenerate it from `E:\kagglegolf`:

```powershell
cd "E:\kongming\NGC-work\workplace C\dashboard"
.\refresh_from_kagglegolf.ps1
```

## Reports

- `reports/task_visibility_and_sources.md`: evidence that the Kaggle data
  package exposes all 400 task JSON files.
- `reports/onnx_visualization_sources.md`: Kaggle discussion evidence for
  ONNX/task visualization tooling.

## Engineering Workspace

- `neurogolf-2026-work/`: Kaggle/NVIDIA skill research scaffold and local
  baseline project.
- `score_docs/34_FULL_RETROSPECTIVE_AND_NEXT_BUILD_PLAN_20260711.md`: current
  score state, accepted/rejected experiments, rebase evidence, quota policy,
  and the next three-track build plan.
- `score_docs/35_DEEP_MODELING_CAMPAIGN_20260712.md`: strict `67/67` individual
  modeling completion, final-ten cost results, and the task298 online-verified
  `7273.42` submission.
- `score_docs/C_DEEP_MODEL_STATUS.md`: per-task evidence checklist for all 67 C
  tasks.
- `cleanup_log.md`: cleanup record for setup-only directories.

The project scaffold still requires `KAGGLE_API_TOKEN` before running Kaggle
API/CLI workflows. The task JSONs, C task viewer, and dashboard do not require
Kaggle credentials.
