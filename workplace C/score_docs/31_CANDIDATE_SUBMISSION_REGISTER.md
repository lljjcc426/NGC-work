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
