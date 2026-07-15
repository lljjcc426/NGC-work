# Submission 5 full-400 repository integration

This folder records a full 400-task integration pass based on the team-provided
`submission (5).zip`. The scan was not limited to the B assignment: every
tracked ONNX file and every model inside every tracked submission ZIP in the
team repository was deduplicated and compared task by task.

## Baseline and package

- Baseline archive SHA256:
  `4DCF46E72FD4890D867A9BB025EDD1E989B9A7978F86198670CE50F04E9A2DA8`.
- Baseline Kaggle ref: `54733776`, public score `7419.90`.
- Integrated package: 400 root-level ONNX files, CRC clean.
- Integrated package SHA256:
  `BCEF5A066B75FC36F231DEC160E34612EDB1E38007775C9E1A9DFD2407196023`.
- Predicted local gain over the exact baseline: `+1.025026570172`.
- Kaggle ref: `54736568`, status `COMPLETE`, public score `7420.93`.
- Confirmed online gain over ref `54733776`: `+1.03`.

## Full repository audit

- Tracked standalone ONNX files: 136.
- Tracked submission ZIP files: 30.
- Unique task/model hashes: 892.
- Unique models different from the working package: 554.
- Task coverage: 400/400.
- Candidates with positive local gain: 4/554.

The four positive historical candidates were all excluded:

| Task | Local gain | Decision |
| --- | ---: | --- |
| `task012` | `+1.009826` | Reject: 69 mismatches in 500 color-permutation fuzz cases. |
| `task023` | `+0.047938` | Reject: the same hash is hidden-unsafe in Kaggle ref `54517546`. |
| `task134` | `+0.140187` | Reject: its mixed online batch did not realize this local gain. |
| `task344` | `+0.145995` | Reject: this rank approximation already failed the hidden distribution. |

Nine unusually slow baseline tasks were isolated behind a 45-second timeout.
Their already-online-validated baseline models were retained and no unscored
candidate was allowed to replace them.

## Accepted overrides

Ten rewrites previously confirmed hidden-safe on Kaggle were rebased onto the
new baseline. A new exact `task295` rewrite precontracts the constant tail and
reuses its selector branch, reducing cost from 377 to 351.

| Task | Baseline cost | New cost | Gain |
| --- | ---: | ---: | ---: |
| `task018` | 24360 | 16679 | `+0.378792` |
| `task023` | 6353 | 6217 | `+0.021640` |
| `task063` | 1706 | 1556 | `+0.092033` |
| `task076` | 12313 | 11832 | `+0.039848` |
| `task101` | 13071 | 11352 | `+0.141002` |
| `task185` | 1682 | 1623 | `+0.035707` |
| `task209` | 7324 | 6374 | `+0.138929` |
| `task270` | 2719 | 2632 | `+0.032520` |
| `task285` | 18189 | 17025 | `+0.066134` |
| `task295` | 377 | 351 | `+0.071459` |
| `task328` | 5189 | 5153 | `+0.006962` |

All 11 replacements pass their complete local train, test, and ARC-GEN sets.

## Contents

- `submission/base_submission.zip`: exact team baseline.
- `submission/submission.zip`: complete 400-task integrated package.
- `models/`: the 11 accepted overrides.
- `reports/`: all-400 inventory, all 554 candidate scores, positive-candidate
  risk list, current scores, and exact baseline-to-current deltas.
- `scripts/`: repository scanner, timeout-safe scorer, incremental score builder,
  deterministic packager, and the reproducible `task295` rewrite.
