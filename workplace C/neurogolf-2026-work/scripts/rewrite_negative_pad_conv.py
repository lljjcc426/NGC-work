from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def node_attributes(node: onnx.NodeProto) -> dict[str, object]:
    return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}


def rewrite_model(model: onnx.ModelProto) -> onnx.ModelProto:
    matches: list[tuple[int, onnx.NodeProto, list[int]]] = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Conv":
            continue
        pads = list(node_attributes(node).get("pads", [0, 0, 0, 0]))
        if len(pads) == 4 and any(value < 0 for value in pads):
            matches.append((index, node, pads))
    if len(matches) != 1:
        raise RuntimeError(f"expected one negative-pad Conv, found {len(matches)}")

    node_index, conv, pads = matches[0]
    top, left, bottom, right = pads
    if top != 0 or left != 0 or bottom >= 0 or right >= 0:
        raise RuntimeError(f"only right/bottom cropping is supported: {pads}")
    crop_h, crop_w = -bottom, -right

    initializers = {item.name: item for item in model.graph.initializer}
    weight_name = conv.input[1]
    if weight_name not in initializers:
        raise RuntimeError("Conv weights must be an initializer")
    weight_tensor = initializers[weight_name]
    old_weight = numpy_helper.to_array(weight_tensor)
    if old_weight.ndim != 4 or old_weight.shape[2:] != (1, 1):
        raise RuntimeError(f"only 1x1 Conv is supported: {old_weight.shape}")

    expanded = np.zeros((*old_weight.shape[:2], 2, 2), dtype=old_weight.dtype)
    expanded[:, :, 0, 0] = old_weight[:, :, 0, 0]
    weight_tensor.CopyFrom(numpy_helper.from_array(expanded, weight_name))

    attrs = node_attributes(conv)
    replacement = helper.make_node(
        "Conv",
        list(conv.input),
        list(conv.output),
        name=conv.name or "safe_dilated_crop",
        dilations=[crop_h, crop_w],
        group=int(attrs.get("group", 1)),
        kernel_shape=[2, 2],
        pads=[0, 0, 0, 0],
        strides=list(attrs.get("strides", [1, 1])),
    )
    nodes = list(model.graph.node)
    nodes[node_index] = replacement
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace one right/bottom negative-pad 1x1 Conv with an exact valid dilated Conv."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = rewrite_model(onnx.load(args.source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
