# E online-positive batch on team 7384.93

This folder records two E-task improvements rebased onto the exact team package
scored at `7384.93` on Kaggle.

## Parent

- GitHub parent commit: `d8a1ca3`.
- Kaggle parent ref: `54686944`.
- Parent public score: `7384.93`.
- Parent ZIP SHA256:
  `8eba05f6379d98c2f99e56760468521d912d23fd0a247556bc1dc2ef2211207`.
- E baseline validation: `67/67` tasks and `16798/16798` examples passed.

## Changed tasks

| Task | Method | Cost | Local gain | Differential fuzz | Isolated Kaggle |
| --- | --- | ---: | ---: | ---: | ---: |
| `task035` | compact scatter plus shared cast-mask fold | `1879 -> 1743` | `+0.075132` | `0/500` mismatches | `7385.00` |
| `task092` | grouped selectors and exact structure compression | `6553 -> 5631` | `+0.151636` | `0/500` mismatches | `7385.08` |

The isolated scores use the same `7384.93` parent. Their displayed online
gains are therefore `+0.07` and `+0.15`, consistent with the local estimates.

## Cumulative Kaggle result

- Submission ref: `54707421`.
- Status: `COMPLETE`.
- Public score: `7385.15`.
- Online gain over the exact parent: `+0.22`.
- Changed models versus the parent: `task035.onnx`, `task092.onnx` only.
- Submission ZIP SHA256:
  `09367f8c6bcc438c3e481178a19c1fba5ffa87f0cf301727caff3925cb801d91`.

## Resume point

Use `submission/submission.zip` as the next E parent. Keep `task012` and
`task233` on hold because differential color-permutation testing found output
mismatches. Continue with a separately isolated, fully validated E task and
record the Kaggle result before accumulating it.
