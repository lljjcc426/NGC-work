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

## 2026-07-10 E scoreboard loop

- Pulled latest `lljjcc426/NGC-work` main branch to `af72da5`; observed B/C/D teammate updates, while `workplace F` still only contained `readme.md` locally.
- Kaggle submissions checked:
  - Current account best visible in the submission list: `7270.18`, timestamp `2026-07-10 07:41:44.663000`, description `v81 F 20plus task006 306 local +1.186266 sha 1dbac62b`.
  - Local search under `F:/kaggle` did not find a zip whose SHA256 starts with `1dbac62b`, so the exact F package is not locally reproducible from this checkout.
- Established the E-team score loop requested by the team:
  - Score all 67 E assignment slots from the current local `submission.zip`.
  - Optimize the lowest-ranked E task.
  - If a candidate verifies on all released examples and lowers cost, update the scoreboard and re-sort.
  - Continue with the new lowest-ranked task.
- Current local E scoreboard:
  - Source zip: `F:/kaggle/neurogolf-2026/submissions/submission.zip`
  - SHA256: `02b3d922f85f7c280e6f2cdf76405dbe3eea29f53d66603b7b4e8fec501a7a69`
  - Output files: `e_scoreboard_20260710.csv`, `e_scoreboard_summary_20260710.json`
  - 67/67 E tasks scored successfully.
  - 8/67 E tasks are above 20 points.
  - Lowest E task: `task233`, cost `31938`, points `14.628448198`.
  - Cost target for >20 points is `148` or lower.
- `task233` source review:
  - Existing local zip/source scan (`e_local_source_scan_20260709.csv`) found no valid lower-cost `task233` replacement versus the yusuke 7267.31 base.
  - Historical experiment scan (`e_task233_experiment_score_20260710.csv`) scored 46 local `task233.onnx` files; 33 verified, but the best verified historical cost was `35021`, worse than current `31938`.
  - Focused yusuke-base optimization scan (`e_task233_yusuke_opt_scan_20260710.csv`) tested conservative onnxoptimizer, patch branch drops, and gate bypasses.
  - Result: `ok_improved=0`. Lower-cost candidates existed but failed validation: best lower-cost rows were `31577 sample_wrong`, `31901 full_wrong`, `31918 sample_wrong`, so no scoreboard update was made for `task233`.
- `task233` rule analysis:
  - Public local rule text in `external/datasets/logic-for-each-arc-task/files/arc_explanations.csv` claims a crop-only rule, but this does not match the released examples.
  - `e_task233_rule_components_20260710.csv` shows `crop_matches_output=0/266`.
  - Every external component found in train/test examples is a `3x3` patch; each patch corresponds to 9 changed cells inside the main crop.
  - Example evidence: train 0 has 5 external components and 45 changed cells; train 1 has 2 external components and 18 changed cells; train 2 has 1 external component and 9 changed cells; test 0 has 4 external components and 36 changed cells.
- A primary-owner branch check:
  - Remote branch inspected: `origin/workplaceA-update-20260709`.
  - It contains `workplaceA/task233.onnx` with Git blob size `20139`.
  - Exported and scored it as a temporary probe; output file `e_task233_A_branch_probe_score_20260710.csv`.
  - Result: verified `ok`, cost `32011`, points about `14.626`; this is worse than current E scoreboard cost `31938`, so no replacement was accepted.
  - Deleted temporary probe files after scoring: `workplace E/task233_from_A_branch_probe.onnx`, `workplace E/probes/task233_A_branch`, and empty `workplace E/probes`.
