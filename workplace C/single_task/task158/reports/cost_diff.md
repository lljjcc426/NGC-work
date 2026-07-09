# task158 Cost Diff

Baseline:

- artifact: `E:\kagglegolf\submissions\candidates\GOLF_20260709_101_prvsiyan_7266_72_repro\onnx\task158.onnx`
- old_cost: `28483`
- old_points: `14.742937302942996`

Candidate:

- artifact: `workplace C\single_task\task158\onnx\task158_candidate.onnx`
- method: `task158_resize_stamp_builder`
- local_valid: `true`
- examples_checked: `266`
- new_cost: `28023`
- new_points: `14.759219119459042`
- delta_cost: `460`
- delta_points: `0.01628181651604521`
- accepted: `true`

Change:

The verified task158 graph already implements source motif detection and scale 1/2/3 square marker-pair stamping. The candidate keeps that rule but replaces the large scale-2 and scale-3 expanded `Gather` index tensors with nearest-neighbor `Resize` from the scale-1 orientable stamp mask. This removes `144 + 324 - 8 = 460` counted parameters while preserving all 266 examples.
