# Workplace D Worklog

Updated: 2026-07-09 13:03:39 +0800

## 2026-07-09 D task optimization scan

- Scope: 67 primary D tasks from `assignments/task_assignment_400.csv`.
- Candidate roots: work/final-v157-stable-plus-task286-on-v154, work/final-v154-stable-no-task325-on-v151, work/final-v147-safe-ryosuke-on-v144, work/public-kernel-outputs, work/latest_public_20260707, work/latest_sources_20260707, work/hand_optimized.
- Raw candidates: 16486; unique candidates: 1034.
- Accepted replacements: 1.
- Aggregate local delta: cost -4, points 0.000756334.
- Full scan CSV: `workplace D/d_candidate_scan_20260709.csv`.
- Accepted CSV: `workplace D/d_accepted_optimizations_20260709.csv`.
- Optimized ONNX folder: `workplace D/optimized_onnx`.

Accepted replacements:

- `task029`: cost 5288 -> 5284 (-4), points +0.000756334; source `work/final-v157-stable-plus-task286-on-v154/task029.onnx`.

## 2026-07-09 Deep full-source candidate scan

- Scope: expanded from the initial hand-picked roots to local `work`, `outputs`, root `submission.zip`, Downloads, and prior `workplace D/optimized_onnx`.
- Deep scan script: `workplace D/d_deep_candidate_scan.py`.
- Deep scan CSV: `workplace D/d_deep_candidate_scan_20260709.csv`.
- Deep accepted CSV: `workplace D/d_deep_accepted_optimizations_20260709.csv`.
- Deep scan report: `workplace D/d_deep_scan_report_20260709.md`.
- Raw candidates: 95,451; unique candidates: 2,077; selected for validation: 1,818.
- Accepted replacements: 8.
- Aggregate local delta after final validation: cost -2,823, points +1.686903354.

Accepted replacements after deep scan:

- `task029`: cost 5288 -> 5284 (-4), points +0.000756334.
- `task055`: cost 3141 -> 1856 (-1285), points +0.526117087.
- `task105`: cost 2941 -> 2866 (-75), points +0.025832392.
- `task115`: cost 1043 -> 1014 (-29), points +0.028197816.
- `task200`: cost 753 -> 623 (-130), points +0.189518481.
- `task256`: cost 2251 -> 2114 (-137), points +0.062792834.
- `task287`: cost 1994 -> 910 (-1084), points +0.784453400.
- `task400`: cost 1181 -> 1102 (-79), points +0.069235010.

## 2026-07-09 Recent 726x source sweep

- Additional check: `workplace D/d_recent_726_source_scan_20260709.csv`.
- Sources swept: `latest_lucifer_agi_core_0709`, `latest_ryosuke_726683`, `latest_ryosuke_726648`, `latest_frank_726672`, `latest_prv_726672`, and `latest_kutenk_726153`.
- Rows validated: 402.
- Result: confirmed the deep-scan winners from recent sources; no ninth D-task winner was found in this focused sweep.

## 2026-07-09 Final optimized ONNX validation

- Validation CSV: `workplace D/d_optimized_validation_20260709.csv`.
- Optimized folder: `workplace D/optimized_onnx`.
- Valid optimized ONNX files: 8 / 8.
- Aggregate validated delta before unscanned sweep: cost -2,823, points +1.686903354.

## 2026-07-09 Unscanned candidate sweep

- Follow-up report: `workplace D/d_unscanned_scan_report_20260709.md`.
- Full sweep CSV: `workplace D/d_unscanned_candidate_scan_20260709.csv`.
- Incremental accepted CSV: `workplace D/d_unscanned_accepted_optimizations_20260709.csv`.
- Final cumulative accepted CSV: `workplace D/d_final_accepted_optimizations_20260709.csv`.
- Unscanned candidates evaluated: 337.
- Valid candidates: 121; timeout / heavy candidates: 16.
- New accepted replacement: `task105`, cost 2866 -> 2849 versus the previous optimized file.
- Final optimized ONNX validation: 8 / 8 valid.
- Final aggregate validated delta: cost -2,840, points +1.692852628.
