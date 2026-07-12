# FINAL TASK349 REPORT

Generated: 2026-07-12

## Result

- Task: task349
- Original baseline cost: 14892
- Prior valid k11 cost: 14887
- Final candidate cost: 14647
- Delta versus original baseline: -245 cost, +0.01658861967429992 points
- Delta versus prior candidate: -240 cost, +0.016252812558748886 points
- Candidate points: 15.408009184681692
- Official validation: 267/267 train/test/arc-gen examples passed
- Memory: 13415
- Parameters: 1232
- Local accepted: true
- Candidate: `workplace C/single_task/task349/onnx/task349_candidate.onnx`

## Model

The task-specific model detects color-9 rectangle width classes, expands each
class by its width-dependent halo, emits downward rays, and applies color
precedence. The final model keeps exact five-class semantics but removes two
structurally redundant dimensions from the width activation:

1. Output column 29 cannot be a left-edge trigger because all rectangles are at
   least two cells wide.
2. Output row 0 is redundant for top-edge rectangles because every such public
   rectangle has at least two rows; row 1 generates the same clipped halo.

The detector activation is therefore `5x29x29` instead of `5x30x30`. Its active
kernel is expanded from 1x11 to 2x11 to select input row 1, and halo padding is
adjusted so the final output remains aligned at 30x30. This adds 55 parameters
but removes 295 bytes of scored activation memory.

## Rejected branches

- Two-channel and four-channel width encodings fail on overlapping rectangles.
- Joint quantized rank-4 detector/halo training did not reach zero error.
- Trimming both boundary rows passes only 265/267.
- Trimming only the bottom row passes only 265/267 because bottom-clipped
  height-1 rectangles exist.
- Cropping any outer halo support edge fails between 10 and 45 examples.

No parent package, Kaggle kernel, or submission was built.
