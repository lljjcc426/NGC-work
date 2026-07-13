# Score Work Report

Generated: 2026-07-09T20:56:21

## Latest Single-Task Accepted Improvement

- task: `task158`
- mode: single-task rule + task-specific ONNX surgery
- Python rule validation: `266/266` examples passed
- old_cost: `28483`
- new_cost: `28023`
- delta_cost: `460`
- old_points: `14.742937302942996`
- new_points: `14.759219119459042`
- delta_points: `0.01628181651604521`
- accepted: `true`
- artifact: `workplace C\single_task\task158\onnx\task158_candidate.onnx`
- builder: `workplace C\single_task\task158\scripts\build_task158_onnx.py`

The accepted change keeps the verified motif-copy rule but replaces scale-2 and scale-3 expanded stamp `Gather` index tensors with nearest-neighbor `Resize` from the scale-1 orientable stamp mask. This removes `460` counted params while preserving full train/test/arc-gen validation.

## Latest Single-Task Rule Model Without Accepted ONNX

- task: `task286`
- mode: single-task rule modeling + ONNX sparse-constant probe
- Python rule validation: `265/265` examples passed
- old_cost: `26909`
- new_cost: `26909`
- delta_cost: `0`
- accepted: `false`
- reason: sparse initializer candidates are rejected by ONNX checker/type inference for the relevant `Conv`, `Where`, `Pad`, and `MatMulInteger` inputs.
- next builder target: rewrite the bitset flood-fill graph itself with fewer scalar bitwise intermediates.

## Modes

- Mode A: ran P0/P1 artifact reuse scan and official full-validation cost scoring.
- Mode B: created reusable artifact scanner, cost diff runner, candidate validator, task card generator, and surgery probe scripts.
- Mode C: summarized local public notebook/bundle intel, centered on the verified prvsiyan 7266.72 baseline.
- Mode D: candidate register updated, but no new candidate package was built because there were zero accepted C replacements.
- Mode E: minimal score docs/task cards generated to support next experiments.

## Direct Score Attempts

- P0/P1 artifacts indexed: `144`.
- Artifact rows full-scored: `73`; accepted: `0`.
- Surgery rows full-scored: `56`; accepted: `0`.
- Generic optimizer/simplifier passes did not reduce official cost; several files became larger on disk but cost stayed identical.
- Single-task task158 rule/surgery: `1` accepted replacement, `28483 -> 28023`.
- Single-task task286 rule/sparse probe: rule solved `265/265`, no accepted ONNX candidate.

## Cost Results

| source | scored_rows | accepted | best_delta_cost |
| --- | ---: | ---: | ---: |
| artifact_scan_top5 | 73 | 0 | 0 |
| onnx_surgery_probe | 56 | 0 | 0 |
| task158_resize_stamp_builder | 1 | 1 | 460 |
| task286_sparse_constant_probe | 1 baseline check + 4 checker probes | 0 | 0 |

## Quick-Win Top 10

| rank | task | priority | current_cost | current_points |
| ---: | --- | --- | ---: | ---: |
| 1 | task158 | P0_lt16 | 28483.0 | 14.742937 |
| 2 | task286 | P0_lt16 | 26909.0 | 14.799784 |
| 3 | task054 | P0_lt16 | 25394.0 | 14.857732 |
| 4 | task349 | P0_lt16 | 14892.0 | 15.391421 |
| 5 | task364 | P0_lt16 | 14642.0 | 15.408351 |
| 6 | task077 | P0_lt16 | 7657.0 | 16.056624 |
| 7 | task009 | P1_16_16p7 | 6694.0 | 16.191033 |
| 8 | task383 | P1_16_16p7 | 5830.0 | 16.329228 |
| 9 | task382 | P1_16_16p7 | 5695.0 | 16.352656 |
| 10 | task096 | P1_16_16p7 | 7678.0 | 16.053886 |

## Next 5 Experiments

1. Build a full 400-file candidate package replacing only `task158.onnx`, then validate `file_count=400` and `missing_task_count=0`.
2. Write a lower-memory row-bitset flood-fill builder for `task286` from the verified BFS rule.
3. Apply the expanded-stamp `Gather -> Resize` rewrite to `task054` if its marker-driven graph uses repeated stamp indices.
4. Mine prvsiyan visualizations and KaggLoop 7266.48 for more stamp-upscale or motif-copy graphs.
5. Only after packaging and quota check, submit the task158 replacement candidate if the user explicitly confirms.

## Blockers

- No full 400-file candidate package has been built yet for the accepted task158 replacement.
- Kaggle submission still requires explicit user confirmation.
- Generic simplification remains ineffective; task-specific graph rewrites are the productive path.
- For `task286`, constant-only sparse initializer surgery is blocked by ONNX checker/type inference; improvement requires a graph rewrite.

## Git

Checkpoint commit created and pushed to `origin/main`:

- commit: `c62c95f`
- message: `Add task158 accepted resize stamp improvement`
- pushed: yes, `5d65954..c62c95f main -> main`

The accepted ONNX artifact remains local and ignored by `.gitignore`; lightweight scripts, reports, debug examples, and score docs were committed.

## 2026-07-12 Deep Modeling Completion

- Strict C-task individual modeling status: `67/67` complete.
- The final ten tasks were rebuilt with independent rule-level graph structures and validated on every public train/test/arc-gen example.
- All final-ten structures were semantically correct but cost-negative, so none were merged.
- Campaign accepted improvement: task298 `135 -> 129`, `267/267`, expected points delta `+0.045462`.
- Exact rebase parent: user upload SHA256 `d3284267c02846dde8571890d4c761dcf9592fce2ec190c3348a0dee1c13c44f`, matching team-best v93 score `7273.37`.
- Submitted candidate: `GOLF_20260712_099_v93_plus_task298`, ref `54595725`.
- Public result: `7273.42`, observed delta `+0.05`; online verification passed.
- Detailed report: `35_DEEP_MODELING_CAMPAIGN_20260712.md`.

## 2026-07-12 Local-Only High-Yield Continuation

The current instruction explicitly excluded parent packages, kernels, and
submissions. Work therefore focused only on official local task cost.

- accepted local replacements: 8
- full-validation failures among accepted artifacts: 0
- combined expected local points gain: `+3.107248641435721`
- largest result: task193 `910 -> 170`, `266/266`, `+1.677646`
- second: task372 `710 -> 360`, `266/266`, `+0.679161`
- third: task230 `900 -> 460`, `266/266`, `+0.671168`
- no parent rebase performed
- no kernel built
- no Kaggle submission performed

All eight artifacts were rescored together with the official local utility and
passed every public train/test/arc-gen example. Full details and rejected
hypotheses are in `36_LOCAL_TASK_SCORE_CAMPAIGN_20260712.md`.

## 2026-07-12 Local Continuation And Verification

- task077 accepted: `7655 -> 7234`, 266/266, `+0.056567` points.
- task096 accepted: `7678 -> 6850`, 266/266, `+0.114110` points.
- task349 accepted: `14647 -> 12480`, 267/267, `+0.160108` points.
- additional local expected gain: `+0.3307855132791033`.
- cumulative gain across the two local passes: `+3.438034154714824`.
- one explicitly requested combination was rebased on the user-provided
  `submission (4).zip`; Kaggle ref `54604279` completed at `7276.61`, matching
  the local expectation from parent score `7273.50`.
- submission management stopped immediately after that result; subsequent work
  returned to the local task list.

## 2026-07-12 Local Task Plan Progress

- submission-related work remained disabled
- six additional accepted task-specific rewrites
- task332: `561 -> 438`, 267/267
- task237: `1836 -> 1716`, 266/266
- task091: `2759 -> 2730`, 266/266
- task009: `6595 -> 6585`, 265/265
- task158: `28023 -> 26250`, 266/266
- task072: `421 -> 368`, 268/268
- additional expected local gain: `+0.5270889313257321`
- cumulative current local campaign gain: `+3.965123086040556`
- detailed report: `38_LOCAL_TASK_PLAN_PROGRESS_20260712.md`

## 2026-07-12 Submission 5 Rebase

- exact parent: `E:/submission (5).zip`, SHA256 `d53db8c5eb5111d065f7fcc241581584da0930a06fa9c15145364bae1c14e47b`
- parent ref/score: `54606556` / `7277.83`
- parent-aware accepted overlays: 22 tasks
- expected gain: `+0.9191029246708331`
- submission ref: `54608730`
- public score: `7278.75`
- observed gain: `+0.92`
- detailed retrospective: `39_SUBMISSION5_REBASE_RETROSPECTIVE_20260712.md`

## 2026-07-13 C+D Archive Research

- source attachment: `E:/archive.zip`, 399 ONNX files, missing task173
- discussion source: Kaggle dataset `zealous9230/neurogolf7300`
- C+D official full benchmark: 134/134 parent models and 133/133 available archive models valid
- archive lower than parent: 54 tasks (30 C, 24 D)
- direct archive replacement: prohibited by local policy; all such rows on compliance hold
- independent accepted improvement: task075 `1487 -> 1394`, 265/265, `+0.0645833551`
- deterministic one-round archive derivation: 6 changed, 2 lower than source and parent
- held candidates: task158 `18530`, task182 `6065`; both `do_not_submit`
- task315 rank-two argmax probe rejected by official per-channel threshold validation
- detailed report: `43_CD_ARCHIVE_METHOD_INTEL_20260713.md`
