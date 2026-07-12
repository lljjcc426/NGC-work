# FINAL TASK307 REPORT

Generated: 2026-07-11

- Parent cost: `5`
- Graph: one `MaxRoiPool`
- Parameters: five ROI coordinates
- Status: minimal-model audit complete; no replacement

The output is produced directly, so there is no counted intermediate tensor.
Replacing ROI crop-and-resize with Slice plus Resize would add an activation and
more parameters. The five-parameter parent is retained.
