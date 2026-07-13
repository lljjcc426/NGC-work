# C Five-Batch Online Verification - 2026-07-13

## Outcome

The C candidates were rebased on the team-accepted `7296.04` package and
submitted as five cumulative Kaggle kernels. All five completed successfully.

- parent score: `7296.04`
- final score: `7297.02`
- observed cumulative gain: `+0.98`
- locally predicted cumulative gain: `+0.9836862798`
- final changed tasks: `23`
- Kaggle submissions used: `5`

No model from `archive.zip`, `neurogolf7300`, or the archive-derived optimizer
was included. Every replacement came from the independent C experiment ledger
and was revalidated against all official train, test, and arc-gen examples
before packaging.

## Submission Sequence

| batch | new tasks | cumulative tasks | expected cumulative gain | Kaggle ref | public score | observed incremental gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | task332 | 1 | +0.247502 | 54633385 | 7296.29 | +0.25 |
| 2 | task349, task091, task383, task388, task094, task190 | 7 | +0.429145 | 54633579 | 7296.47 | +0.18 |
| 3 | task072, task077 | 9 | +0.620523 | 54633759 | 7296.66 | +0.19 |
| 4 | task096, task075, task392, task381, task378 | 14 | +0.802180 | 54633899 | 7296.84 | +0.18 |
| 5 | task158, task237, task009, task224, task046, task382, task165, task132, task364 | 23 | +0.983686 | 54634057 | 7297.02 | +0.18 |

## Package Hashes

| batch | SHA-256 |
| --- | --- |
| 1 | `F57CF24492CB78F21D2339F61B3B4DFEE163E85C802FEF6283B0197717E1CD0E` |
| 2 | `6F3943880C7583F4DC9DB1DB66C316C5DC89031E81419196C41DC15D7B62E555` |
| 3 | `3643BFB4F5FC46E4B32E2BD2C0FD92515970FAD69B6E2BAB4EE990A351046FC1` |
| 4 | `B66950E8AC1D755A770F912663F5271A5165F5749FF06988B5ADFAA4C2EA9DCB` |
| 5 | `5374C6BB7A0276FE778F6E6B14996301A47786EDDFD3480122AA91719D42C091` |

The package files remain under `E:/kagglegolf/submissions/candidates/` and are
not committed to GitHub.

## Reproduction

Preparation and official full-example comparison:

```powershell
python "workplace C/neurogolf-2026-work/scripts/c_prepare_five_submit_batches.py" `
  --base-zip "workplace B/20260713_b20_tail_online_v3/submission/accepted_7296.04.zip" `
  --ledger "workplace C/score_docs/30_SCORE_EXPERIMENT_LEDGER.csv" `
  --output-root "E:/kagglegolf/submissions/five_batches_20260713" `
  --workers 3
```

The generated `batch_1_overrides.csv` through `batch_5_overrides.csv` were
built with `scripts/13_single_task_override.py` and submitted with
`scripts/19_submit_queue.py`. Results were refreshed with
`scripts/20_poll_submission_results.py`.

