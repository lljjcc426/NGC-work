# FINAL TASK009 REPORT

Generated: 2026-07-11

## Result

- Public validation: `266/266`
- Parent cost: `6694`
- Candidate cost: `6595`
- Delta cost: `99`
- Parent points: `16.191033118228727`
- Candidate points: `16.205932934848583`
- Delta points: `+0.014899816619855955`
- Status: local accepted; not submitted

## Accepted Design

The baseline decoded each sampled coarse cell into a scalar color and then used
a full `10x10` validity mask plus `Where` to replace padded cells with a
sentinel. The accepted candidate moves that distinction into the first Conv:

- Conv bias is `10`.
- Channel weights are `[-10, -9, ..., -1]`.
- A valid background cell remains `0`.
- Valid colors remain `1..9`.
- All-zero padded input becomes sentinel `10`.

This lets `content_u8` serve directly as `content_grid`, deleting one full
`10x10` uint8 intermediate while retaining a separate outside mask for the
separator line path.

## Rejected Probe

The uint8 Einsum probe passed ONNX checker but ONNX Runtime has no implementation
for uint8 Einsum in the official execution path. It is not usable.

## Artifact

Local artifact: `workplace C/single_task/task009/onnx/task009_outside_sentinel.onnx`

The ONNX file is intentionally excluded from GitHub.
