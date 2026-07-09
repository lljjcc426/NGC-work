# ONNX Visualization Source Notes

Updated: 2026-07-09

## Kaggle Discussion Evidence

The following public NeuroGolf discussion threads were queried with Kaggle CLI:

```powershell
kaggle competitions topics show neurogolf-2026 699313 --format json --page-size 200
kaggle competitions topics show neurogolf-2026 699429 --format json --page-size 200
```

Findings:

- Discussion `699313`, **Web GUI to Build ONNX by Hand**, by Chris Deotte, is an ONNX hand-building GUI thread. A reply links to discussion `699429` as an open-source version.
- Discussion `699429`, **Web GUI for Hand Solving Tasks Open Source**, by Clark Kitchen, is the open-source GUI thread.
- A comment in `699429` flags that `/api/export` can upload ONNX artifacts to Hugging Face if configured. The author replied that the app was fixed to verify the active HF token owner before export.

## Deployment Decision

For this repository we do not vendor or run the external GUI directly. The safe default is a read-only dashboard:

- no Kaggle token access
- no Hugging Face token access
- no ONNX upload endpoint
- no raw competition data or submission zip committed
- only task status, score, cost, source, and assignment metadata are exported

The dashboard is generated from the local `E:\kagglegolf` task table after scoring the reproduced `prvsiyan` 7266.72 artifact.

## Local Dashboard

Open:

```text
workplace C/dashboard/index.html
```

or serve it locally:

```powershell
cd "workplace C\dashboard"
.\serve_dashboard.ps1
```

The HTML auto-refreshes every 30 seconds. To refresh data, regenerate the export from the local `kagglegolf` workspace.
