# Local Optimization Closeout - 2026-07-15

## Baseline and online sequence

- Starting verified parent for this closeout: Kaggle ref `54729703`, public `7410.67`, local `7410.514164178757`.
- Ref `54729003`: COMPLETE `7410.55`; exact terminal factors for `task002`, `task224`, and `task249`.
- Ref `54729703`: COMPLETE `7410.67`; exact transpose deduplication for `task108` and terminal pair precontraction for `task217` and `task275`.
- Ref `54730760`: round44 upload containing exact constant-component contractions for `task075` and `task383`; status and score are recorded below after Kaggle processing.

## Exact improvements closed in this upload

| Task | Exact method | Parent cost | Candidate cost | Local points gain | Official | Exact fuzz |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| task002 | factor terminal border pair | 5888 | 5808 | 0.013681 | 268/268 | 2000/2000 |
| task108 | deduplicate transposed Einsum initializer | 144 | 136 | 0.057158 | 266/266 | 2000/2000 |
| task217 | precontract private terminal Einsum axis | 459 | 442 | 0.037740 | 266/266 | 2000/2000 |
| task224 | factor terminal mix axis | 1436 | 1332 | 0.075225 | 266/266 | exact identity |
| task249 | deduplicate transposed Einsum initializer | 128 | 120 | 0.064539 | 266/266 | 2000/2000 |
| task275 | precontract private terminal Einsum axis | 671 | 659 | 0.018046 | 266/266 | 2000/2000 |
| task075 | precontract constant terminal component | 559 | 554 | 0.008985 | 265/265 | 2000/2000 |
| task383 | precontract constant terminal component | 2593 | 2530 | 0.024596 | 266/266 | 2000/2000 |

## Round44 package

- Replacements: `task075`, `task383`.
- Local predicted gain over ref `54729703`: `+0.0335809902249138`.
- Package SHA256: `e5fe58794ed48ad90330c961b779b1a375a6c69046b54375e45086908225dab8`.
- Validation: `400/400` models and `101347/101347` official examples.
- Validation mode: 398 models inherited by exact parent SHA; both changed models fully revalidated.
- Kaggle ref: `54730760`.
- Kaggle status: `PENDING` at upload closeout.
- Kaggle public score: pending.

## Preserved local work and negative evidence

- Exact factorization and contraction builders are retained as reusable source scripts.
- Single-task builders and evidence for tasks `001`, `003`, `074`, `086`, `110`, `173`, `187`, `224`, `243`, `267`, and `286` are retained.
- Failed or unsafe variants remain evidence only and are not promoted solely because they reduce local cost.
- The task251 hidden-distribution regression and runtime-risk restrictions remain blocked in existing registries and online-result records.
- ONNX models, submission ZIP files, score caches, raw data, credentials, and tokens are intentionally excluded from GitHub.

## Upload boundary

This closeout stops optimization work. It records and uploads the current local source, structured evidence, and ledger state without starting another model search.
