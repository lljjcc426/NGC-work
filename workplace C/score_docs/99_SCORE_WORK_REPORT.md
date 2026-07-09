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
