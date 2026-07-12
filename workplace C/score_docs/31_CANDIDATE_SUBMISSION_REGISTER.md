# Candidate Submission Register

Generated: 2026-07-09T20:56:21

| candidate_id | file_count_400 | missing_task_count | expected_delta_cost | public_lb | status |
| --- | ---: | ---: | ---: | ---: | --- |
| GOLF_20260709_101_prvsiyan_7266_72_repro | 400 | 0 | baseline | 7266.72 | current best source package |
| TASK158_RESIZE_STAMP_20260709 | not built | not checked | +460 cost reduction on task158 |  | single-task replacement accepted locally |

## Accepted Single-Task Replacement

| task | old_cost | new_cost | delta_cost | local_valid | artifact_path | package_status |
| --- | ---: | ---: | ---: | --- | --- | --- |
| task158 | 28483 | 28023 | 460 | true | `workplace C\single_task\task158\onnx\task158_candidate.onnx` | replacement ready; full 400-file package not built in this single-task pass |

Next packaging requirement: copy the current best 400 ONNX files, replace only `task158.onnx`, verify `file_count_400=400` and `missing_task_count=0`, then submit only after explicit user confirmation.

## 2026-07-12 Current Cumulative Candidate

| candidate_id | parent | file_count | missing | cumulative replacements | expected public delta | kernel | submitted |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| GOLF_20260712_098_v97_plus_task298 | v92 / public 7271.95 lineage | 400 | 0 | 20 | +0.09225703817750024 | not built | no |
| GOLF_20260712_099_v93_plus_task298 | v93 / public 7273.37, sha d3284267 | 400 | 0 | 1 | +0.045462374076757 | complete: 7273.42 | yes |

Local path: `E:/kagglegolf/submissions/candidates/GOLF_20260712_098_v97_plus_task298`

Online-verified path: `E:/kagglegolf/submissions/candidates/GOLF_20260712_099_v93_plus_task298`

Submission ref: `54595725`; observed public delta: `+0.05`.

Newest accepted replacement:

| task | old_cost | new_cost | delta points | full validation | artifact |
| --- | ---: | ---: | ---: | --- | --- |
| task298 | 135 | 129 | +0.045462374076757 | 267/267 | `workplace C/single_task/task298/onnx/task298_fused_strip.onnx` |

The candidate remains local. Kernel construction is deferred until Kaggle
quota is near the previously specified five-percent reserve threshold.

## 2026-07-12 Local Replacements Only

Per the current instruction, these are independent local artifacts only. They
have not been rebased, packaged, or submitted.

| task | old cost | new cost | delta points | validation | artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| task193 | 910 | 170 | +1.677646 | 266/266 | `workplace C/single_task/task193/onnx/task193_candidate.onnx` |
| task230 | 900 | 460 | +0.671168 | 266/266 | `workplace C/single_task/task230/onnx/task230_candidate.onnx` |
| task372 | 710 | 360 | +0.679161 | 266/266 | `workplace C/single_task/task372/onnx/task372_candidate.onnx` |
| task349 | 14892 | 14647 | +0.016589 | 267/267 | `workplace C/single_task/task349/onnx/task349_candidate.onnx` |
| task335 | 1380 | 1324 | +0.041426 | 266/266 | `workplace C/single_task/task335/onnx/task335_exact_rank4_template.onnx` |
| task286 | 26909 | 26879 | +0.001115 | 265/265 | `workplace C/single_task/task286/onnx/task286_conv_support_crop.onnx` |
| task069 | 2946 | 2916 | +0.010236 | 264/264 | `workplace C/single_task/task069/onnx/task069_compact_pad_axes_conv_crop.onnx` |
| task201 | 3043 | 3013 | +0.009908 | 266/266 | `workplace C/single_task/task201/onnx/task201_compact_pad_axes_conv_crop.onnx` |
