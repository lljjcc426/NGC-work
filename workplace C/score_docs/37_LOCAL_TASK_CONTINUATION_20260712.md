# C Local Task Continuation - 2026-07-12

## Accepted

| task | structure | old cost | new cost | delta points | validation |
| --- | --- | ---: | ---: | ---: | ---: |
| task077 | binary vertical threshold fusion | 7655 | 7234 | +0.056566896 | 266/266 |
| task096 | compact 19x19 projection Conv | 7678 | 6850 | +0.114110444 | 266/266 |
| task349 | collision-safe three-channel shared halo | 14647 | 12480 | +0.160108173 | 267/267 |

This continuation adds `+0.3307855132791033` expected local points. Together
with the preceding pass, the current unstacked C improvements total
`+3.438034154714824` points.

## Technical Notes

- task077 removes the full-grid `T=2R` path by adjusting the vertical quantized
  kernel so every positive response is exactly one and can be compared to `R`.
- task096 preserves exact counts but projects only the active 19x19 support,
  avoiding two 10x30 float tensors and their larger boolean descendants.
- task349 merges five width classes into three collision-safe channels and uses
  one shared halo Conv. A cheaper two-channel model passes only 192/267 and is
  rejected.

## Public Verification Snapshot

At the user's explicit request, the previous eight-task local stack was rebased
onto `E:/submission (4).zip`. The parent was Kaggle ref `54603337` at 7273.50.
The combined submission ref `54604279` completed at `7276.61`, matching the
expected `+3.108584` delta to two decimal places.

No further Kaggle submission management is part of this continuation.
