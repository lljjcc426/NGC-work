# Workplace D Unscanned Candidate Sweep

Updated: 2026-07-09 20:57:00 +0800

## Scope

- Purpose: validate unique D-task candidates not covered by the prior deep scan.
- Input baseline scan: `workplace D/d_deep_candidate_scan_20260709.csv`.
- Output scan CSV: `workplace D/d_unscanned_candidate_scan_20260709.csv`.
- Accepted incremental CSV: `workplace D/d_unscanned_accepted_optimizations_20260709.csv`.
- Final cumulative accepted CSV: `workplace D/d_final_accepted_optimizations_20260709.csv`.

## Result

- Unscanned unique candidates evaluated: 337.
- Valid candidates: 121.
- Timeout / heavy candidates: 16.
- Candidates beating assignment baseline: 6.
- Candidates beating the current `optimized_onnx` set: 1.

Accepted incremental replacement:

- `task105`: cost `2866 -> 2849` versus the previous optimized file, and `2941 -> 2849` versus assignment baseline; points delta versus baseline `+0.031781666`; source `work/final-v170-anas-on-v169/task105.onnx`.

## Final Current Pack

- Optimized ONNX files: 8.
- Final validation CSV: `workplace D/d_optimized_validation_20260709.csv`.
- Final cumulative delta: cost `-2840`, points `+1.692852628`.

Final accepted replacements:

- `task029`: cost `5288 -> 5284`, points `+0.000756334`.
- `task055`: cost `3141 -> 1856`, points `+0.526117087`.
- `task105`: cost `2941 -> 2849`, points `+0.031781666`.
- `task115`: cost `1043 -> 1014`, points `+0.028197816`.
- `task200`: cost `753 -> 623`, points `+0.189518481`.
- `task256`: cost `2251 -> 2114`, points `+0.062792834`.
- `task287`: cost `1994 -> 910`, points `+0.784453400`.
- `task400`: cost `1181 -> 1102`, points `+0.069235010`.
