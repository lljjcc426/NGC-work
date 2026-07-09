# Workplace C Dashboard

This folder contains a read-only task progress dashboard for Workplace C.

Open `index.html` directly, or run:

```powershell
.\serve_dashboard.ps1
```

Files:

- `index.html`: searchable/filterable C task dashboard with 30-second browser auto-refresh.
- `task_progress_C.csv`: C task status table.
- `task_progress_C.md`: Markdown summary and top gap tasks.
- `source_task_scoreboard_all_tasks.csv`: source all-task scoreboard used for the C export.
- `dashboard_metadata.json`: generation metadata.

This dashboard does not contain Kaggle credentials, Hugging Face credentials, ONNX files, or submission archives.

To regenerate from the local `E:\kagglegolf` workspace:

```powershell
.\refresh_from_kagglegolf.ps1
```
