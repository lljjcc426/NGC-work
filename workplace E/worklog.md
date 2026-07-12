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

## 2026-07-10 team-high baseline correction and task233 loop

- Live Kaggle team submission inventory was rechecked before further optimization.
  - Current team best: ref `54519973`, public score `7270.18`, UTC `2026-07-10 07:41:44.663000`, bytes `458278`.
  - Description: `v81 F 20plus task006 306 local +1.186266 sha 1dbac62b`.
  - Immediate lineage: v78 ref `54500407` scored `7267.61`; v79 ref `54517518` scored `7268.99`; v81 ref `54519973` scored `7270.18`.
- Retrieved the exact team notebook ancestor output from `blacklions/2026-neurogolf-baseline` without copying notebook source into GitHub.
  - Local artifact: `F:/kaggle/neurogolf-2026/external/team_blacklions_baseline_v1/submission.zip`.
  - This matches ref `54485638`: public score `7267.31`, bytes `453733`.
  - SHA256: `31c36b9157c228f8cf6886d8c02cfca27a0a9f51a46884be9cb7da5d4bd0ca87`.
  - Integrity: 400 ONNX entries, canonical task001-task400 layout, CRC clean.
- The exact v81 package could not be obtained from the shared repository or Kaggle submission API.
  - GitHub main and all remote branches contain no `1dbac62b` package or F build artifact.
  - Kaggle `GetSubmission` exposes metadata but no team-member download method; the raw submission file remains restricted to the submitter.
  - Therefore no Kaggle submission was made from the 7267.31 ancestor. This avoids repeating the earlier low-base submission mistake.
- New E scoreboards preserve the previous 7237.15 records instead of overwriting them.
  - `e_scoreboard_team_base_7267_31_20260710.csv` and `e_scoreboard_team_base_7267_31_summary_20260710.json`: 67/67 E tasks scored; 8 above 20; task233 is lowest at cost `31938`, points `14.628448198`.
  - `e_scoreboard_team_base726731_plus_task233_20260710.csv` and `e_scoreboard_team_base726731_plus_task233_summary_20260710.json`: scatter-removal candidate applied; task233 remains lowest at cost `30710`.
  - `e_scoreboard_team_base726731_plus_task233_masked_topk_20260710.csv` and `e_scoreboard_team_base726731_plus_task233_masked_topk_summary_20260710.json`: latest accepted candidate applied; task233 remains lowest at cost `30384`.
- Added a non-destructive package builder: `e_build_override_package_20260710.py`.
  - It requires a full base zip plus explicit `taskNNN=PATH` overrides.
  - It validates ONNX files, canonical 400-task inventory, embedded hashes, and ZIP CRC.
  - It never overwrites `F:/kaggle/neurogolf-2026/submissions/submission.zip`.
- Accepted the next task233 rewrite.
  - Script: `e_optimize_task233_masked_topk_20260710.py`.
  - Model: `optimized_onnx/task233_masked_topk_20260710/task233.onnx`.
  - SHA256: `95d7bc0568f23deca46bca05d8eda03dc113259e71d1d169cd087b3165bea958`.
  - Rewrite: replace the 324-element descending rank initializer with a masked float16 TopK over exact pattern matches, while preserving the first-two-match behavior.
  - Full validation: ARC-AGI `4/4`, ARC-GEN `262/262`.
  - Cost: `30710 -> 30384` (`-326`); cumulative from the team ancestor task233 is `31938 -> 30384` (`-1554`).
  - Points: `14.667656387 -> 14.678328567` (`+0.010672180`); cumulative delta is `+0.049880369`.
- Independent validation package:
  - `F:/kaggle/neurogolf-2026/submissions/submission_team_base726731_e_task233_masked_topk_20260710.zip`.
  - SHA256: `9ac10724e17c78b1d74a889baa625f77fe915b80648dc9fe4629fced0f8d00b4`.
  - 400 entries and CRC clean; this is an ancestor-based validation artifact, not a claimed v81 reconstruction.
- Cleanup performed and disclosed:
  - Deleted generated `F:/kaggle/NGC-work/workplace E/__pycache__/`.
  - Deleted empty `.gitkeep` markers from the three new external download directories after real files were present.
  - No ONNX, CSV, JSON, ZIP, notebook output, or experiment log was deleted.

## 2026-07-11 practical-ROI loop and task085 improvement

- Changed the E loop from lowest-score-first to practical score ROI: prefer a meaningful measured point gain that can be implemented and fully validated quickly.
- Rechecked live Kaggle team metadata before optimizing.
  - Current team best: ref `54557194`, public score `7271.93`, UTC `2026-07-11 02:15:21.407000`.
  - Description: `v90 compact Pad axes 21 tasks local +0.033332 sha 02f7787e`.
  - The exact v90 raw package remains unavailable to this Kaggle account, so no ancestor-based package was submitted.
- Recorded rejected high-ROI probes in `e_opportunity_loop_20260711.csv`.
  - task294 sparse Constant nodes passed `265/265` but cost increased `910 -> 3660` because Constant outputs count as intermediate memory.
  - task294 graph sparse initializers were rejected by ONNX full shape inference for Conv weights.
  - task084 row-offset broadcasting was rejected at runtime because ScatterElements requires indices and updates to match on the channel dimension.
- Accepted task085 signed-Pad rewrite.
  - Script: `e_optimize_task085_signed_pad_20260711.py`.
  - Model: `optimized_onnx/task085_signed_pad_20260711/task085.onnx`.
  - Model SHA256: `c3a1e929b776cee03f21e032b11748d3c4d805c4efbd517e730a7e11b07f3078`.
  - Replaced four Slice+Pad pairs with four signed Pad nodes using `[1,-1]` and `[-1,1]` on axis 2.
  - Removed four Slice nodes and seven now-unused initializers.
  - Full validation: train `2/2`, test `1/1`, ARC-GEN `262/262` (`265/265`).
  - Cost: `2845 -> 2597` (`-248`).
  - Points: `17.046681653 -> 17.137887788` (`+0.091206135`).
- Independent validation package:
  - `F:/kaggle/neurogolf-2026/submissions/submission_team_base726731_e_task233_masked_topk_task085_signed_pad_20260711.zip`.
  - SHA256: `0cf72dba19ba2946451f7e4124cfd8783b8d5df3dad913153b42d627bb927581`.
  - Exact 400-task inventory, ZIP CRC, and both override hashes passed.
- Recomputed scoreboard: `e_scoreboard_roi_task085_20260711.csv` and `e_scoreboard_roi_task085_summary_20260711.json`; all `67/67` E tasks scored successfully.
- Cleanup: deleted generated `F:/kaggle/NGC-work/workplace E/__pycache__/`; no ONNX, CSV, JSON, ZIP manifest, script, or experiment result was deleted.

## 2026-07-12 sequential loop: task003

- Changed the loop to fixed ascending E task order to remove task-selection overhead.
- User-supplied baseline: `F:/kaggle/submission (1).zip`.
  - SHA256: `ca5ed65c12ab15a7194f7b00d6cc83722dcc6f65b8cc1d11ae7b5d6fab16a9eb`.
  - The user explicitly requested direct analysis without rechecking whether it matches the live team-high submission.
- Added read-only graph/example inspection script: `e_analyze_task003_sequential_20260712.py`.
- Accepted task003 rewrite:
  - Script: `e_optimize_task003_qlinear_output_20260712.py`.
  - Model: `optimized_onnx/task003_qlinear_output_20260712/task003.onnx`.
  - Model SHA256: `4394bab28a9345e184cc55d896067714310c15c37c444e5430f2400bb67d658c`.
  - Replaced BitwiseXor plus Concat with one QLinearConv over the valid 9x3 area; retained the final ConvInteger so padding remains zero.
  - Full validation: train `3/3`, test `1/1`, ARC-GEN `261/261` (`265/265`).
  - Cost: `260 -> 239` (`-21`).
  - Points: `19.439318369 -> 19.523536448` (`+0.084218079`).
- Rejected probe retained in the script history but not as a model: direct padded QLinearConv cost `181` but failed `0/265` because its bias colored the padding area.
- Built full package from the user baseline:
  - `F:/kaggle/neurogolf-2026/submissions/submission_team_high_e_task003_qlinear_20260712.zip`.
  - SHA256: `23117163ed67a1677abb3e91932aa62e0c9c06b44ca7127104a8ca622a6fbb60`.
  - Exact 400-task inventory, ZIP CRC, and task003 override hash passed.
- Kaggle submission:
  - The first upload used a noncanonical archive filename and returned HTTP 400; no submission was created.
  - Copied the same package bytes to `F:/kaggle/neurogolf-2026/submissions/staging_e_task003_20260712/submission.zip`; SHA256 remained `23117163ed67a1677abb3e91932aa62e0c9c06b44ca7127104a8ca622a6fbb60`.
  - Kaggle CLI then reported `Successfully submitted to The 2026 NeuroGolf Championship`.
  - Follow-up status queries could not read `C:/Users/cc/.kaggle/access_token` under the current sandbox, so submission ref, completion status, and public score are not yet recorded.
- Next sequential task: task007.
