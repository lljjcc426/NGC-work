# Score First Start Here

Generated: 2026-07-09T15:58:38

Use this workspace for C group ONNX-equivalent compression experiments only. Do not commit `.onnx`, token, zip, or Kaggle output files.

## Fast Commands

```powershell
python "workplace C\neurogolf-2026-work\scripts\c_quick_win_scan.py"
python "workplace C\neurogolf-2026-work\scripts\c_score_scan_artifacts.py" --tasks P0P1 --score-top-n 5 --full-validate
python "workplace C\neurogolf-2026-work\scripts\c_onnx_surgery_probe.py" --tasks P0P1 --strategies optimizer,sim,optimizer_sim --full-validate
python "workplace C\neurogolf-2026-work\scripts\c_validate_candidate.py" --candidate-dir "E:\kagglegolf\submissions\candidates\GOLF_20260709_101_prvsiyan_7266_72_repro\onnx"
```
