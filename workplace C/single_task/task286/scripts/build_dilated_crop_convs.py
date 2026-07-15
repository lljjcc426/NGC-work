from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _attrs(node: onnx.NodeProto) -> dict[str, object]:
    return {item.name: helper.get_attribute_value(item) for item in node.attribute}


def build(parent_path: Path, output_path: Path) -> Path:
    model = onnx.load(parent_path)
    initializers = {item.name: item for item in model.graph.initializer}
    replacements = 0
    for index, node in enumerate(list(model.graph.node)):
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        weight = initializers.get(node.input[1])
        if weight is None:
            continue
        value = numpy_helper.to_array(weight)
        if value.ndim != 4 or value.shape[2:] != (6, 6):
            continue
        if np.any(value[:, :, 0, 1:] != 0) or np.any(value[:, :, 1:, :] != 0):
            raise RuntimeError(f"{node.input[1]} has support outside the top-left corner")

        compact = np.zeros((*value.shape[:2], 2, 2), dtype=value.dtype)
        compact[:, :, 0, 0] = value[:, :, 0, 0]
        weight.CopyFrom(numpy_helper.from_array(compact, name=node.input[1]))
        attrs = _attrs(node)
        if attrs.get("auto_pad", b"NOTSET") not in {b"NOTSET", "NOTSET"}:
            raise RuntimeError("auto_pad is not supported")
        pads = list(attrs.get("pads", [0, 0, 0, 0]))
        strides = list(attrs.get("strides", [1, 1]))
        if pads != [0, 0, 0, 0] or strides != [1, 1]:
            raise RuntimeError(f"unexpected Conv geometry: pads={pads}, strides={strides}")
        replacement = helper.make_node(
            "Conv",
            list(node.input),
            list(node.output),
            name=node.name or f"task286_dilated_crop_{replacements}",
            dilations=[5, 5],
            group=int(attrs.get("group", 1)),
            kernel_shape=[2, 2],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
        )
        model.graph.node[index].CopyFrom(replacement)
        replacements += 1

    if replacements != 2:
        raise RuntimeError(f"expected two 6x6 crop Conv nodes, rewrote {replacements}")
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(inferred, output_path)
    reloaded = onnx.load(output_path)
    onnx.checker.check_model(reloaded, full_check=True)
    onnx.shape_inference.infer_shapes(reloaded, strict_mode=True, data_prop=True)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Replace task286's two sparse 6x6 crop Conv kernels with exact dilated 2x2 kernels.")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.parent, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
