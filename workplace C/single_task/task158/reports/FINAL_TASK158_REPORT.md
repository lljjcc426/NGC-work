# FINAL TASK158 REPORT

## Result

- task: `task158`
- Python rule analysis complete: `yes`
- Python rule solver: `workplace C/single_task/task158/scripts/solve_task158_rule.py`
- train pass: `3/3`
- test pass: `1/1`
- arc-gen pass: `262/262`
- total pass: `266/266`
- ONNX candidate generated: `yes`
- candidate artifact: `workplace C/single_task/task158/onnx/task158_pair3_candidate.onnx`
- accepted: `yes`

## Cost

| old_cost | new_cost | delta_cost | old_points | new_points | delta_points |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 28483 | 26250 | 2233 | 14.742937302942996 | 14.82457873198023 | 0.08164142903723359 |

## Rule

The task is a same-shape sparse overlay problem. Each input has a dominant background, a 3x3 motif containing two opposite-corner marker colors and one fill color, and one or more square marker-color target pairs. The output preserves the input and overlays a rotated/reflected/scaled copy of the motif's fill mask between each compatible marker pair. Output colors are input-palette-only.

## ONNX Change

The baseline ONNX was already task-specific. The accepted candidate keeps the source-motif and marker-pair logic, but removes the scale-2 and scale-3 expanded stamp index tensors:

- removed `stamp_idx2` with `144` params
- removed `stamp_idx3` with `324` params
- added `stamp_size2` and `stamp_size3` with `8` params total
- replaced both `Gather` stamp builders with nearest `Resize` from `stamp_w1`

The second accepted rewrite encodes four orientation states in three
collision-safe channels. A small 1x1 quantized transform reconstructs the
three signed stamp channels, and all scale-specific pair detectors are reduced
from four channels to three. Relative to the first accepted candidate this
reduces official cost from 28023 to 26250 and remains 266/266.

## Validation Command

```powershell
python "workplace C\neurogolf-2026-work\scripts\c_cost_diff_runner.py" --task task158 --old-artifact "E:\kagglegolf\submissions\candidates\GOLF_20260709_101_prvsiyan_7266_72_repro\onnx\task158.onnx" --new-artifact "workplace C\single_task\task158\onnx\task158_candidate.onnx" --method task158_resize_stamp_builder
```

## Next

Try sharing the remaining scale-specific pair kernels or folding the 1x1
orientation transform into the first scale convolution. No submission work is
part of this local task pass.
