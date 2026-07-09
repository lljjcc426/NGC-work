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

## 2026-07-09 public 7266 source scan

- Kaggle public metadata checked with `kaggle kernels list -s neurogolf --sort-by dateRun --page-size 20`.
- Downloaded public notebook outputs locally under `F:/kaggle/neurogolf-2026/external/source_review_20260709_e/`; these external files are not committed to GitHub.
- Public sources scored for E tasks:
  - `franksunp/7266-72-lb-compact-onnx-artifact-starter`, `is_private=false`
  - `prvsiyan/neurogolf-7266-72-w-visualizations`, `is_private=false`
  - `ryosukeshiroshita/neurogolf-7266-48-github-com-qurore-kaggloop`, `is_private=false`
  - `seddiktrk/neurogolf-2026-all-graph-surgeries`, `is_private=false`
- Output scan: `e_public_7266_source_scan_20260709.csv`.
  - Rows: 268 source-task checks.
  - Improved checks: 62.
- Selected manifest: `e_public7266_selected_manifest_20260709.csv`.
  - Selected 21 E assignment tasks.
  - All selected models came from `franksunp_7266_72`.
  - Released local delta sum: `+4.931264`.
- Built local candidate archive:
  - `F:/kaggle/neurogolf-2026/submissions/submission_e_public7266_union_20260709.zip`
  - SHA256: `0bccf7d2cdf8184f293703ff6b074a1db8cfb441b1c1a6bc71c7061a6b9f1f83`
  - Zip integrity: 400 entries, no missing tasks, no extra entries, `testzip=None`.
- First Kaggle submit attempt used the nonstandard filename `submission_e_public7266_union_20260709.zip` and returned HTTP 400.
- Copied candidate archive to standard Kaggle entrypoint `F:/kaggle/neurogolf-2026/submissions/submission.zip`.
  - Previous `submission.zip` did not exist.
  - Overwrite/copy record: `submission_zip_overwrite_20260709.json`.
- Kaggle submission:
  - UTC timestamp: `2026-07-09 04:59:51.290000`
  - Description: `E public7266 union 21 tasks local +4.931 sha 0bccf7d2`
  - Status: `COMPLETE`
  - publicScore: `7255.20`
  - Previous best in submission list: `7250.28`
  - Verified leaderboard delta: `+4.92`
