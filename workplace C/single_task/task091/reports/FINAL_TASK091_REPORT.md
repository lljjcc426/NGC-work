# FINAL TASK091 REPORT

- Rule model: gray-guide bbox crop with cyan overlay
- Validation: `266/266`
- Parent cost: `2759`
- Best cost: `2730`
- Delta points: `+0.010566686`
- Status: accepted locally

The direct gray-row projection was rejected because it increased memory. The
accepted rewrite keeps the scalar endpoint decoder and upgrades only the Pad
path to opset 18, replacing an eight-axis dynamic pads vector with the four
spatial pads that are actually consumed.

Artifact: `workplace C/single_task/task091/onnx/task091_candidate.onnx`.
