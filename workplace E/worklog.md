# E Team Worklog

## 2026-07-09

- Pulled latest `lljjcc426/NGC-work` main branch.
- Confirmed E has 67 assignment slots: 66 primary tasks plus shared review for `task233`.
- Constraint: do not use closed-source Kaggle notebook code. Public Kaggle sources require source/license/reproducibility notes before use.
- Local public-source scan produced `e_candidate_scan_20260709.csv`; result: 0 E tasks had a lower-cost already-scored local public/source candidate than current v244.
- Added `e_optimize_equiv_scan.py` to test conservative equivalence optimization against current `submission_v244_union_20260708.zip`.
- Ran `python "workplace E\e_optimize_equiv_scan.py"`.
  - Output: `e_equiv_opt_scan_20260709.csv`.
  - Result: 67/67 E assignment slots checked, 0 safe cost improvements.
- Manual probe on `task085`:
  - Released-data rule observed: middle row of each 3-row stripe clears every second colored cell from the stripe left edge.
  - Current v244 model is already only `Conv -> Cast -> Where`, cost `5381`.
  - Tried fp16 Conv variant: full released validation passed, but cost worsened from `5381` to `21581`.
  - Deleted temporary probe model `workplace E/task085_fp16_probe.onnx`; keep current v244 model.
