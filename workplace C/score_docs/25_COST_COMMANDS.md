# Cost Commands

Generated: 2026-07-09T15:58:38

Primary official scoring path:

```powershell
python "workplace C\neurogolf-2026-work\scripts\c_score_scan_artifacts.py" --tasks P0P1 --score-top-n 5 --full-validate
python "workplace C\neurogolf-2026-work\scripts\c_cost_diff_runner.py" --task task158 --old-artifact <old.onnx> --new-artifact <new.onnx> --method <method>
```

The scripts import `E:/kagglegolf/data/raw/neurogolf-2026/neurogolf_utils/neurogolf_utils.py` and compute `cost=memory+params` after local example validation.
