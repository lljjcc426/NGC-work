# Workplace D Deep Candidate Scan

Updated: 2026-07-09 16:20:58 +0800

## Scope

- D primary tasks: 67.
- Source roots: workplace D/optimized_onnx, work, outputs, submission.zip, ..
- Raw candidates: 95451; unique candidates: 2077; selected for validation: 1818.
- Selection budget: `D_DEEP_MAX_PER_TASK=40`, `D_DEEP_MAX_BYTES=16000`.

## Accepted Replacements

- Accepted replacements: 8.
- Aggregate local delta: cost -2823, points 1.686903354.

- `task029`: cost 5288 -> 5284 (-4), points +0.000756334; source `work/final-v155-legacy-pass2-on-v154/task029.onnx`.
- `task055`: cost 3141 -> 1856 (-1285), points +0.526117087; source `work/public-kernel-outputs/latest_lucifer_agi_core_0709/submission.zip::task055.onnx`.
- `task105`: cost 2941 -> 2866 (-75), points +0.025832392; source `work/public-kernel-outputs/latest_lucifer_agi_core_0709/submission.zip::task105.onnx`.
- `task115`: cost 1043 -> 1014 (-29), points +0.028197816; source `work/public-kernel-outputs/latest_lucifer_agi_core_0709/submission.zip::task115.onnx`.
- `task200`: cost 753 -> 623 (-130), points +0.189518481; source `work/public-kernel-outputs/latest_lucifer_agi_core_0709/submission.zip::task200.onnx`.
- `task256`: cost 2251 -> 2114 (-137), points +0.062792834; source `work/public-kernel-outputs/latest_lucifer_agi_core_0709/submission.zip::task256.onnx`.
- `task287`: cost 1994 -> 910 (-1084), points +0.784453400; source `work/public-kernel-outputs/latest_lucifer_agi_core_0709/submission.zip::task287.onnx`.
- `task400`: cost 1181 -> 1102 (-79), points +0.069235010; source `work/public-kernel-outputs/latest_ryosuke_726683/submission.zip::task400.onnx`.

## Outputs

- Full scan CSV: `workplace D/d_deep_candidate_scan_20260709.csv`.
- Accepted CSV: `workplace D/d_deep_accepted_optimizations_20260709.csv`.
- Optimized ONNX folder: `workplace D/optimized_onnx`.
