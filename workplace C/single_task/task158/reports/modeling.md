# task158 independent modeling

The task overlays a rotated or reflected motif between compatible marker
pairs while preserving the same-size input. The Python solver identifies the
motif, marker pair, scale, and fill color without reading expected outputs.

The dedicated ONNX keeps sparse overlay logic but replaces scale-2 and scale-3
expanded Gather index banks with nearest-neighbor Resize from the scale-1
stamp. It passes all 266 examples and lowers cost from 28483 to 28023.
