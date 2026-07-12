from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def attribute(node: onnx.NodeProto, name: str, default):
    return next((helper.get_attribute_value(item) for item in node.attribute if item.name == name), default)


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    conv = model.graph.node[0]
    if conv.op_type != "Conv":
        raise RuntimeError("first node is not Conv")
    weight = next(item for item in model.graph.initializer if item.name == conv.input[1])
    array = numpy_helper.to_array(weight)
    support = np.argwhere(np.any(array != 0, axis=(0, 1)))
    if support.size == 0 or not np.array_equal(support.min(0), [0, 0]) or not np.array_equal(support.max(0), [0, 0]):
        raise RuntimeError("Conv support is not exactly the top-left 1x1 cell")
    dilation = list(attribute(conv, "dilations", [1, 1]))
    pads = list(attribute(conv, "pads", [0, 0, 0, 0]))
    removed_h = dilation[0] * (array.shape[2] - 1)
    removed_w = dilation[1] * (array.shape[3] - 1)
    compact = array[:, :, :1, :1]
    weight.CopyFrom(numpy_helper.from_array(compact, name=weight.name))
    del conv.attribute[:]
    conv.attribute.extend([
        helper.make_attribute("kernel_shape", [1, 1]),
        helper.make_attribute("pads", [pads[0], pads[1], pads[2] - removed_h, pads[3] - removed_w]),
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, output)
    return output
