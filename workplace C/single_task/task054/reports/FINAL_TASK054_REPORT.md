# FINAL TASK054 REPORT

## Scope

Worker scope was limited to `E:/kongming/NGC-work/workplace C/single_task/task054/`.
No Kaggle submission was made. No parent package was touched.

## Baseline

- Baseline artifact: `E:/kagglegolf/submissions/candidates/GOLF_20260709_101_prvsiyan_7266_72_repro/onnx/task054.onnx`
- Full local validation: `266/266`
- Memory: `25201`
- Params: `193`
- Cost: `25394`
- Points: `14.857731795370263`

## Rule Summary

task054 is a same-shape template projection task. The input contains one small template object and usually two large rectangular objects. The baseline graph finds the small template, identifies marker cells in large rectangles, projects horizontal/vertical frontiers from the template into the large rectangles, then overlays the local template neighborhood around each marker.

## Attempts

### Attempt 1: final scalar ScatterElements fusion

Artifact: `onnx/task054_fusion_scatter_none.onnx`

Idea: replace the final `ScatterElements -> Greater -> Where` line overlay with a direct scalar `ScatterElements` into the final canvas.

Result:

- Full validation: `0/266`
- Memory: `23761`
- Params: `193`
- Cost: `23954`
- Status: invalid

Failure reason: the original graph uses `ScatterElements` to build an OR-style line mask, then excludes marker cells with `Greater(scatter_mask, marker_mask)`. Direct scalar scatter does not preserve that mask semantics.

### Attempt 2: final scalar ScatterElements fusion with reduction=max

Artifact: `onnx/task054_fusion_scatter_max.onnx`

Idea: keep the direct scalar scatter but use `reduction=max` to partially mimic duplicate-index OR behavior.

Result:

- Full validation: `10/266`
- Memory: `23761`
- Params: `193`
- Cost: `23954`
- Status: invalid

Failure reason: `max` is not equivalent to overwrite semantics when line color is not monotonically larger than the existing color.

### Attempt 3: int32 overlay index compression

Artifact: `onnx/task054_index32_overlay.onnx`

Idea: keep the baseline output fusion unchanged and reduce the final `4x8x4` neighbor overlay index path from int64 to int32.

Result:

- Full validation: not runnable
- Status: invalid graph

Failure reason: the official ONNX Runtime path rejects int32 indices for `GatherND` in this model. Bumping opset did not make this variant usable without rewriting other ops whose attributes changed in newer opsets.

## Conclusion

No accepted positive task054 replacement was produced in this worker pass.

The most promising low-risk output-fusion idea lowered nominal cost from `25394` to `23954`, but it failed correctness. The failure is structural: task054 needs line-mask OR semantics plus marker-cell exclusion before final overlay. Removing the final `Greater + Where` is not locally safe unless the line mask is rebuilt as a true bool mask without adding another full-grid tensor.

## Next Best Action

Do not submit any task054 artifact from this pass.

If task054 is revisited, the next concrete route is a larger reorder:

1. Build combined line mask including marker cells.
2. Apply line overlay before local template-neighborhood overlay.
3. Let the existing local template `ScatterND` restore marker/neighborhood cells.
4. Only continue if this deletes one full-grid tensor rather than replacing it with another.
