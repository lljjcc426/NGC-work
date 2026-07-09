# Current C Score Work State

Generated: 2026-07-09T15:58:38

- C group tasks identified: 67 primary tasks.
- P0/P1 tasks identified: 14 (`task158, task286, task054, task364, task349, task077, task096, task009, task383, task382, task278, task165, task378, task132`).
- P0/P1 task cards: 67 generated; P0/P1 cards include full validate actions and concrete experiments.
- ONNX artifacts indexed: `144` P0/P1 unique local artifacts.
- Officially scored artifact candidates: `73`; accepted: `0`.
- ONNX surgery probes: `56` rows; accepted: `0`.
- Cost command exists: `c_score_scan_artifacts.py` and `c_cost_diff_runner.py` use local official `neurogolf_utils.py`.
- Validator exists: `c_validate_candidate.py` checks 400-task candidate completeness.
- Public notebook baseline: `GOLF_20260709_101_prvsiyan_7266_72_repro` public LB `7266.72`.
- Current highest-yield next route: write dedicated compact builders for task158/task286/task054/task364 instead of generic graph cleanup.
