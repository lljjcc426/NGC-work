# ALL399 Direct Batch Results - 2026-07-13

## Scope

- Parent package: `GOLF_20260713_C5_05`
- Parent public score: `7297.02`
- Public source: `SRC_KAGGLE_DATASET_ZEALOUS_NEUROGOLF7300`
- Source scan: 399 ONNX files; `task173` absent
- Direct replacements: 128 source artifacts that were locally valid and lower cost than the parent
- Local-only fillers: `task009`, `task377`
- Total unique replacements: 130
- Packaging: 13 cumulative batches, exactly 10 new tasks per batch
- Validation: every candidate contained 400 ONNX files and passed the local candidate validator

## Batch Plan And Online State

| Batch | New tasks | Expected cumulative score | Submission ref | Observed score/status |
|---:|---|---:|---:|---|
| 01 | 045,078,043,060,188,052,033,332,310,084 | 7317.920029 | 54636646 | 7317.92 COMPLETE |
| 02 | 058,287,304,362,079,041,099,114,257,273 | 7331.885854 | 54636773 | 7331.89 COMPLETE |
| 03 | 296,240,132,225,195,272,254,322,348,075 | 7342.260190 | 54636908 | 7342.26 COMPLETE |
| 04 | 040,237,389,139,202,298,211,207,220,072 | 7350.178312 | 54637116 | 7350.18 COMPLETE |
| 05 | 066,199,093,108,373,355,100,167,105,089 | 7356.997724 | 54637278 | 7357.00 COMPLETE |
| 06 | 252,370,331,342,034,028,251,157,073,321 | 7362.551910 | 54637436 | 7362.56 COMPLETE |
| 07 | 095,388,203,246,383,005,378,288,294,036 | 7367.173625 | 54637603 | 7367.18 COMPLETE |
| 07 duplicate | same as batch 07 | 7367.173625 | 54638561 | PENDING at last check |
| 08 | 130,152,365,358,190,374,392,006,262,334 | 7370.995953 | 54638565 | PENDING at last check |
| 09 | 158,118,198,027,345,382,330,335,394,050 | 7373.995925 | 54638567 | PENDING at last check |
| 10 | 021,183,253,290,064,263,351,030,284,020 | 7376.235789 | 54638571 | PENDING at last check |
| 11 | 393,224,367,091,156,379,229,356,354,065 | 7377.536973 | 54638574 | PENDING at last check |
| 12 | 258,231,285,171,316,357,144,150,242,037 | 7377.961307 | 54638577 | PENDING at last check |
| 13 | 054,122,049,209,059,101,208,243,009,377 | 7377.991144 | 54638601 | PENDING at last check |

## Execution Notes

1. Batches 01-06 were submitted sequentially with result polling. Every completed score matched the local projection to leaderboard display precision.
2. Batch 04 used the locally validated ZIP fallback after notebook output retrieval failed. The resulting score still matched the projection.
3. The original sequential loop submitted batch 07 before it was terminated. A second batch 07 submission was then created during the no-wait bulk upload. This duplicate consumes one quota slot but does not change the cumulative package content.
4. Kernels 08-13 were uploaded concurrently and all reached `KernelWorkerStatus.COMPLETE` before their competition submissions were issued.
5. Competition submissions 08-13 were accepted by the CLI and assigned refs. Per user instruction, their public scores were not polled before this report was written.

## Evidence Boundary

- `claimed_score` from a dataset title or description is not treated as verified.
- The only verified online values in this report are the Kaggle submission history scores listed as `COMPLETE`.
- Source artifacts are tracked as public-dataset provenance with `risk=high`; this report does not assert that every source artifact is independently authored by the dataset publisher.
- No raw data, token, cookie, `submission.zip`, or ONNX artifact is committed to GitHub.

## Next Check

Run the existing submission-history query once after Kaggle finishes refs `54638565` through `54638601`. Attribute each observed cumulative delta against the preceding batch; do not count ref `54638561` as a new model change.
