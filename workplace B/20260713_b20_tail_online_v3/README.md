# B tail integration: 7296.04

This folder continues from the accepted `7290.38` package. Twelve additional
B tasks were integrated in four threshold-sized submissions. Every submission
completed online with the predicted gain direction and magnitude.

| Added tasks | Kaggle ref | Score | Online gain |
| --- | ---: | ---: | ---: |
| task244, task377 | 54628864 | 7292.17 | +1.79 |
| task024, task245 | 54628977 | 7293.55 | +1.38 |
| task291, task368, task369 | 54629107 | 7294.97 | +1.42 |
| task001, task143, task255, task313, task344 | 54629233 | 7296.04 | +1.07 |

The complete run from `7282.01` to `7296.04` adds `+14.03` online. The final
400-model submission has SHA-256
`4CD81F2341673942CB95F8F959DCAFA6D194BB0A948B524DC38CD104CE165A20`.

## Validation notes

- task377 was reverse engineered as consecutive duplicate row/column removal.
  In addition to all official examples, it passed 500 independently generated
  legal concentric grids with compressed sizes 3, 5, 7, and 9.
- task024 is a zero-memory terminal Einsum for the full-row/full-column rule.
- task291 is a zero-memory terminal Einsum with cost 40.
- task001 preserves the required 3x3 Kronecker coordinate ordering; a cheaper
  ordinary flattened outer product was rejected because it permutes blocks.
- task285 is deliberately absent. Its no-Pad shortcut is known to fail fresh
  generator inputs and must not be merged into this package.

The next independent rewrite target is task255. Even after this integration it
has cost 5976 and only 16.304493 points, making it one of the weakest B tasks.

