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
- candidate artifact: `workplace C/single_task/task158/onnx/task158_candidate.onnx`
- accepted: `yes`

## Cost

| old_cost | new_cost | delta_cost | old_points | new_points | delta_points |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 28483 | 28023 | 460 | 14.742937302942996 | 14.759219119459042 | 0.01628181651604521 |

## Rule

The task is a same-shape sparse overlay problem. Each input has a dominant background, a 3x3 motif containing two opposite-corner marker colors and one fill color, and one or more square marker-color target pairs. The output preserves the input and overlays a rotated/reflected/scaled copy of the motif's fill mask between each compatible marker pair. Output colors are input-palette-only.

## ONNX Change

The baseline ONNX was already task-specific. The accepted candidate keeps the source-motif and marker-pair logic, but removes the scale-2 and scale-3 expanded stamp index tensors:

- removed `stamp_idx2` with `144` params
- removed `stamp_idx3` with `324` params
- added `stamp_size2` and `stamp_size3` with `8` params total
- replaced both `Gather` stamp builders with nearest `Resize` from `stamp_w1`

Net official cost reduction: `460`.

## Validation Command

```powershell
python "workplace C\neurogolf-2026-work\scripts\c_cost_diff_runner.py" --task task158 --old-artifact "E:\kagglegolf\submissions\candidates\GOLF_20260709_101_prvsiyan_7266_72_repro\onnx\task158.onnx" --new-artifact "workplace C\single_task\task158\onnx\task158_candidate.onnx" --method task158_resize_stamp_builder
```

## Next

1. Build a full 400-file candidate package by copying current best ONNX files and replacing only `task158.onnx`.
2. Validate package file count equals `400` and missing task count equals `0`.
3. If submission quota remains and user explicitly confirms, submit the package.
4. Try the same `Gather expanded stamp -> Resize` rewrite on other motif/stamp tasks, starting with `task286` and `task054`.
