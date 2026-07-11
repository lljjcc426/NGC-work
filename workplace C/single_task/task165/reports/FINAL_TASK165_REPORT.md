# FINAL TASK165 REPORT

Generated: 2026-07-11

## Result

- Public validation: `265/265`
- Parent cost: `4532`
- Candidate cost: `4528`
- Delta cost: `4`
- Delta points: `+0.00088300226487803`
- Status: local accepted; not submitted

## Accepted Design

The model was upgraded from opset 13 to opset 18. Its only full-rank Pad
constant was replaced by compact pads plus an axis input. Four ReduceMax nodes
share two small axis initializers. Official scoring reports a net four-parameter
reduction with unchanged activation memory.

## Artifact

Local artifact: `workplace C/single_task/task165/onnx/task165_compact_pad_axes.onnx`

The ONNX file is excluded from GitHub.
