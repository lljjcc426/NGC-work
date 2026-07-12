# Zealous9230 neurogolf7300+ B-task audit

Source: `https://www.kaggle.com/datasets/zealous9230/neurogolf7300`

This is a task-level audit, not a blind merge. The Kaggle dataset contains 399
ONNX files and omits `task173.onnx`. Filling that one gap from our current
submission gives a locally valid 400-task score of `7371.855070`, but that does
not establish hidden-generator safety. In particular, the dataset's smaller
`task285` is the already-rejected no-Pad shortcut that fails fresh ARC-GEN
inputs at Gather index 924.

## Accepted B overrides

| Task | Cost change | Local gain | Validation | Online result |
| --- | ---: | ---: | --- | --- |
| task293 | 1043 -> 40 | +3.260977 | 267 official/ARC-GEN cases | isolated probe: 7280.93 |
| task056 | 34 -> 30 | +0.125163 | 46 official/ARC-GEN cases | accepted in aggregate |
| task104 | 238 -> 118 | +0.701586 | all official cases | accepted in aggregate |
| task205 | 2691 -> 2084 | +0.255624 | all official cases | accepted in aggregate |

The second probe combines `task056`, `task104`, and `task205` on top of the
accepted task293 package. Its predicted gain was `+1.082373`; Kaggle reported
`7282.01`, exactly `+1.08` over the task293 probe and `+4.34` over the previous
`7277.67` baseline.

## What we learned

- `task293` compiles the crossing-line swap into one parameter-40 terminal
  high-order Einsum. It uses self-weighted polynomial contractions instead of
  materializing masks and branches.
- `task104` compiles the complete rule into one parameter-118 terminal Einsum.
  Repeating the input as Einsum operands eliminates the Slice intermediate and
  beats our earlier rank-2 sign model at cost 160.
- `task056` reads only the required corners with two GatherElements operations,
  decodes the boolean choice, and Pads the result. The old Slice/Not/And/cast
  chain is unnecessary.
- A high local aggregate score is not sufficient evidence. Each aggressive
  construction must be isolated, checked against its exact generator, and then
  probed online when the accumulated gain exceeds one point.

`reports/b_positive_comparison.json` contains all 25 local B improvements for
future one-task-at-a-time reverse engineering. The next high-value targets are
`task161`, `task163`, and `task212`; `task350` has the largest remaining raw
gain but its large terminal Einsum needs a stronger hidden-safety audit first.

