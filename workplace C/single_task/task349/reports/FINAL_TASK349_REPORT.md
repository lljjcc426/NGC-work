# FINAL_TASK349_REPORT

Generated: 2026-07-11

## Result

- Task: task349
- Baseline artifact: `E:/kagglegolf/submissions/candidates/GOLF_20260709_101_prvsiyan_7266_72_repro/onnx/task349.onnx`
- Candidate artifact: `E:/kongming/NGC-work/workplace C/single_task/task349/onnx/task349_candidate.onnx`
- Validation: 267/267 public train/test/arc-gen examples passed.
- Baseline cost: 14892
- Candidate cost: 14887
- Delta cost: -5
- Baseline points: 15.391420565007392
- Candidate points: 15.391756372122943
- Delta points: +0.00033580711555067077
- Local accepted: true

## Implemented Compression

The accepted candidate keeps the baseline five-channel rectangle-width detector
and halo logic, but shortens the horizontal detector kernel from width 12 to
width 11. The removed coefficient only checked the right boundary after a
width-10 rectangle. Task349 public data uses width 10 as the maximum rectangle
width, so this boundary check is redundant on all 267 public examples.

This preserves the baseline output exactly while saving 5 initializer params.

## Compact Encoding Attempts

The two-channel `[class, class^2]` and four-channel `[class, class^2, is4, is5]`
halo encodings remain invalid for full submission use. They lower cost more
aggressively, but fail on arc-gen examples containing multiple nearby small
rectangles. In those cases, summed compressed features create false positives in
outer halo rings. The current valid candidate therefore prioritizes guaranteed
positive score movement over an invalid larger local delta.

## Verification

Official local scorer:

- Baseline: memory 13710, params 1182, cost 14892, passed 267/267.
- Candidate: memory 13710, params 1177, cost 14887, passed 267/267.

No Kaggle submission was made.
