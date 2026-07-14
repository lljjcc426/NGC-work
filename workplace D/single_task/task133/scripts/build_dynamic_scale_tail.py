from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _initializer(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value, name=name)


def _drop_unused_initializers(model: onnx.ModelProto) -> None:
    used = {name for node in model.graph.node for name in node.input if name}
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def build(source: Path, output: Path) -> None:
    model = onnx.load(source)
    nodes = list(model.graph.node)
    if len(nodes) < 134 or nodes[95].output[0] != "pat" or nodes[107].output[0] != "idx4":
        raise RuntimeError("unexpected task133 parent graph")

    prefix = nodes[:96] + nodes[99:108]
    tail = [
        helper.make_node("Greater", ["scode", "c0"], ["dyn_valid"], name="dyn_valid"),
        helper.make_node(
            "Where", ["dyn_valid", "H", "c0"], ["dyn_selected_scale"], name="dyn_selected_scale"
        ),
        helper.make_node(
            "Where", ["dyn_valid", "scode", "c0"], ["dyn_update_f"], name="dyn_update_f"
        ),
        helper.make_node(
            "Cast", ["dyn_update_f"], ["dyn_update_u8"], name="dyn_update_u8", to=onnx.TensorProto.UINT8
        ),
        helper.make_node("Reshape", ["dyn_update_u8", "sh10"], ["dyn_update"], name="dyn_update"),
        helper.make_node(
            "ScatterND", ["zbase", "idx4", "dyn_update"], ["dyn_seed"], name="dyn_seed", reduction="add"
        ),
        helper.make_node(
            "ReduceMax",
            ["dyn_selected_scale", "dyn_axes_all"],
            ["dyn_scale_f"],
            name="dyn_scale_f",
            keepdims=0,
        ),
        helper.make_node(
            "Cast", ["dyn_scale_f"], ["dyn_scale"], name="dyn_scale", to=onnx.TensorProto.INT64
        ),
        helper.make_node("Mul", ["dyn_scale", "dyn_hw_base"], ["dyn_hw"], name="dyn_hw"),
        helper.make_node(
            "Resize",
            ["pat", "", "", "dyn_hw"],
            ["dyn_kernel"],
            name="dyn_kernel",
            coordinate_transformation_mode="asymmetric",
            mode="nearest",
            nearest_mode="floor",
            axes=[2, 3],
        ),
        helper.make_node("Mul", ["dyn_scale", "dyn_pad_mult"], ["dyn_pad_raw"], name="dyn_pad_raw"),
        helper.make_node("Sub", ["dyn_pad_raw", "dyn_pad_sub"], ["dyn_pads"], name="dyn_pads"),
        helper.make_node(
            "Pad", ["dyn_seed", "dyn_pads", "", "ax23"], ["dyn_padded"], name="dyn_padded", mode="constant"
        ),
        helper.make_node(
            "QLinearConv",
            ["dyn_padded", "onef", "qzp", "dyn_kernel", "onef", "qzp", "onef", "qzp"],
            ["dyn_stamp"],
            name="dyn_stamp",
        ),
        helper.make_node("Max", ["grid", "dyn_stamp"], ["outgrid"], name="dyn_outgrid"),
        nodes[133],
    ]
    del model.graph.node[:]
    model.graph.node.extend(prefix + tail)
    model.graph.initializer.extend(
        [
            _initializer("dyn_axes_all", np.asarray([0, 1, 2, 3], dtype=np.int64)),
            _initializer("dyn_hw_base", np.asarray([4, 6], dtype=np.int64)),
            _initializer("dyn_pad_mult", np.asarray([3, 4, 1, 2], dtype=np.int64)),
            _initializer("dyn_pad_sub", np.asarray([1, 1, 0, 0], dtype=np.int64)),
        ]
    )
    _drop_unused_initializers(model)
    model.producer_name = "ngc_task133_dynamic_scale_tail"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
