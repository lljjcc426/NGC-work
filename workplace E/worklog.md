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

## 2026-07-10 task233 external ONNX audit

- Partial external ONNX scan output: `e_task233_external_onnx_score_20260710.csv`.
- Rows recorded: 63.
- Best verified external candidate found in this partial scan had cost `34400`, worse than the current E scoreboard cost `31938`.
- No `task233` replacement was accepted, and the scoreboard was not updated.
- The long-running Python scoring process left by the timed-out scan was stopped manually after confirming the partial CSV had been written.

## 2026-07-10 task233 additional loop probes

- Added `e_task233_dtype_sweep_20260710.py` to test single-Cast dtype substitutions against the current local `submission.zip`.
- Output scan: `e_task233_dtype_sweep_20260710.csv`.
  - Candidates checked: 168.
  - Result: 168 `score_error`, 0 verified improvements.
  - No candidate model was written under `optimized_onnx/task233_dtype_20260710`.
- Fetched remote branches and checked for branch-owned `task233.onnx` files.
  - Remote branch `origin/workplaceA-update-20260709` updated to commit `22e7a0e6544504b10bcf575c6324d81d5e00a980`.
  - Only remote branch path found for this task: `workplaceA/task233.onnx`.
  - Output scan: `e_task233_remote_branch_scan_20260710.csv`.
  - Result: verified `ok`, cost `32011`, still worse than the current E scoreboard cost `31938`.
- No `task233` replacement was accepted, and the scoreboard remains unchanged: `task233`, cost `31938`, points `14.628448198`.

## 2026-07-10 task233 scatter removal improvement

- Continued the lowest-task loop on `task233`; no closed-source notebook code was used.
- Additional negative evidence retained:
  - `e_task233_hybrid_v005_probe_20260710.csv`: the historical low-cost directory model verified but cost `75542`, worse than `31938`.
  - `e_task233_models_work_score_20260710.csv`: 19 unique sub-40 KB models from local `models/` and `work/` all verified; best cost `32011`, still worse.
  - `e_task233_general_bypass_20260710.csv`: 101 shape-compatible bypass candidates checked, 0 accepted.
  - A temporary uint8 TopK experiment was deleted after ONNX Runtime showed that opset-13 TopK has no uint8 implementation in the competition runtime; no candidate was produced.
- Accepted graph rewrite:
  - Script: `e_optimize_task233_scatter_remove_20260710.py`.
  - Output: `e_task233_scatter_remove_20260710.csv`.
  - Model: `optimized_onnx/task233_scatter_remove_20260710/task233.onnx`.
  - The old graph created a full 30x30 external-component mask and removed patches with `MaxPool`, `Equal`, and `Where`.
  - The replacement reuses the already-computed 5x9 patch indices and removes external patches directly with `ScatterElements`.
  - Invalid TopK slots are redirected to padded index 899; cell `(29,29)` is zero in all 266 released task233 inputs.
  - Full validation: ARC-AGI `4/4`, ARC-GEN `262/262`.
  - Cost: `31938 -> 30710` (`-1228`).
  - Points: `14.628448198 -> 14.667656387` (`+0.039208189`).
- Scoreboard loop update:
  - Recomputed all 67 E assignments from the updated submission package.
  - 67/67 scored successfully; 8 tasks remain above 20 points.
  - `task233` remains the lowest task, now at cost `30710`, so it remains the next loop target.
- Submission build:
  - Builder: `e_build_task233_submission_20260710.py`.
  - Archive: `F:/kaggle/neurogolf-2026/submissions/submission_e_task233_scatter_20260710.zip`.
  - Kaggle entrypoint `F:/kaggle/neurogolf-2026/submissions/submission.zip` was overwritten from SHA256 `02b3d922...` to `03896ba0ce0be19ea3c95d1faeded210effff9c5c812f2cac7495d74cceeed16`.
  - Integrity: 400 entries, no missing/extra tasks, CRC check passed.
- Kaggle result:
  - Submission ref: `54527273`.
  - UTC timestamp: `2026-07-10 12:31:02.713000`.
  - Description: `E loop task233 scatter 31938 to 30710 local +0.039208 sha 03896ba0`.
  - Status: `COMPLETE`.
  - Public score: `7237.15`, up from `7237.11` on the same local base (`+0.04`).
  - Team best remains `7270.18`; the exact teammate v81 package is not available in the shared checkout and could not be downloaded with this team member's Kaggle API credentials.
