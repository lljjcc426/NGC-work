# B-20 task104 rank-2 sign rewrite v1

This folder contains an independently derived B-only rewrite of task104.

## Result

- Cost: `238 -> 160`.
- Points: `19.527729 -> 19.924826`.
- Gain: `+0.397097`.
- Current gap to 20: reduce another 12 cost units (`160 -> 148`).
- Validation: all 7 official examples, covering all four legal orientations.
- Threshold output: exact for `7/7` examples.
- Minimum positive raw margin: `5.0`.

## Method

The original graph stores three spatial Einsum components. The ARC rule has
exactly one active corner orientation, and colors 0 and 3 are complementary on
the generated 9x9 output. A numerical sign-factor search followed by integer
rounding found a two-component indefinite inner product with vectors from
`{(3,-2), (-3,-2), (0,-3)}`. It reproduces all four legal orientation masks
with a sign margin of at least 5.

The final graph keeps the four-value corner Slice, replaces the 180-value
spatial factor and 30-value color factor with a 120-value rank-2 factor, a
2-value metric, and one 10-value shared color direction.

Use `model/task104.onnx` as the task-level override. The gain is below the
`+1.0` package submission threshold, so it is retained for the next aggregate.

The `experiments/` directory contains rejected task395 constructions only.
They are negative evidence and must not be submitted.
