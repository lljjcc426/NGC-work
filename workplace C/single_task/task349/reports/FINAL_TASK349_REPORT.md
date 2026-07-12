# FINAL TASK349 REPORT

- Model: collision-safe three-channel width encoding with one shared halo convolution.
- Full official validation: `267/267`, zero mismatches.
- Starting candidate: `memory=13415`, `params=1232`, `cost=14647`.
- Final candidate: `memory=11733`, `params=747`, `cost=12480`.
- Cost delta: `-2167`.
- Points: `15.4080091847 -> 15.5681173581` (`+0.1601081734`).
- Accepted locally: **yes** (`local_valid=true`, `12480 < 14647`).
- Artifact: `workplace C/single_task/task349/onnx/task349_candidate.onnx`.
- Builder: `workplace C/single_task/task349/scripts/build_double_collision_merge.py`.
- Rejected lower-cost branch: two channels, cost 11396, only `192/267`.
- Parent package, Kaggle kernel, and Kaggle submission: none.
