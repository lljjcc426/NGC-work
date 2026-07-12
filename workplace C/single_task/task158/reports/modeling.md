# task158 independent modeling

The task overlays a rotated or reflected motif between compatible marker
pairs while preserving the same-size input. The Python solver identifies the
motif, marker pair, scale, and fill color without reading expected outputs.

The dedicated ONNX keeps sparse overlay logic but replaces scale-2 and scale-3
expanded Gather index banks with nearest-neighbor Resize from the scale-1
stamp. It passes all 266 examples and lowers cost from 28483 to 28023. A second
task-specific rewrite compresses the four orientation channels into three and
lowers cost further to 26250, also passing 266/266.
