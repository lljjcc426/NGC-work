# FINAL TASK383 REPORT

Generated: 2026-07-11

## Result

- Public validation: `266/266`
- Parent cost: `5830`
- Candidate cost: `5800`
- Delta cost: `30`
- Delta points: `+0.005159082810031634`
- Status: local accepted; not submitted

## Accepted Design

The first Conv used a `2x2` kernel with dilation 6, but only the top-left
coefficient was nonzero for every input channel. The zero coefficients existed
only to crop the output from `30x30` to `24x24`.

The candidate replaces it with a `1x1` Conv and negative bottom/right padding
`-6`. Sampling and output shape remain identical while the weight tensor drops
from 40 to 10 parameters.

## Artifact

Local artifact: `workplace C/single_task/task383/onnx/task383_conv_crop_collapse.onnx`

The ONNX file is excluded from GitHub.
