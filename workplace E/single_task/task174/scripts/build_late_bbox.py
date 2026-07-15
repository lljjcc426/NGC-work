from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


REMOVED_OUTPUTS = {
    "rx",
    "bu",
    "bot",
    "rxr",
    "tu",
    "trev",
    "top",
    "rl",
    "W_i",
    "bt",
    "H_i",
    "top_b",
    "left_b",
    "Hb",
    "Wb",
}


def _reshape_index_initializers(model: onnx.ModelProto) -> None:
    for name in ("xidxf", "xrevf"):
        index = next(
            (i for i, item in enumerate(model.graph.initializer) if item.name == name),
            None,
        )
        if index is None:
            raise RuntimeError(f"missing initializer {name}")
        array = numpy_helper.to_array(model.graph.initializer[index])
        if array.shape == (10,):
            continue
        if array.shape != (1, 1, 10):
            raise RuntimeError(f"unexpected {name} shape: {array.shape}")
        model.graph.initializer[index].CopyFrom(
            numpy_helper.from_array(np.asarray(array).reshape(1, 10), name)
        )


def build(parent_path: Path, output_path: Path) -> None:
    model = onnx.load(parent_path)
    _reshape_index_initializers(model)
    nodes = list(model.graph.node)
    removed = {
        index
        for index, node in enumerate(nodes)
        if any(output in REMOVED_OUTPUTS for output in node.output)
    }
    expected_types = {
        "rx": "Mul",
        "bu": "ReduceMax",
        "bot": "Cast",
        "rxr": "Mul",
        "tu": "ReduceMax",
        "trev": "Cast",
        "top": "Sub",
        "rl": "Sub",
        "W_i": "Add",
        "bt": "Sub",
        "H_i": "Add",
        "top_b": "Gather",
        "left_b": "Gather",
        "Hb": "Gather",
        "Wb": "Gather",
    }
    producers = {output: node.op_type for node in nodes for output in node.output}
    if any(producers.get(name) != op_type for name, op_type in expected_types.items()):
        raise RuntimeError("unexpected task174 bbox chain")

    rewritten = [node for index, node in enumerate(nodes) if index not in removed]
    pick_index = next(
        index for index, node in enumerate(rewritten) if "pick_index" in node.output
    )
    replacement = [
        helper.make_node(
            "Gather", ["left", "pick_index"], ["left_b"], name="pick_left", axis=0
        ),
        helper.make_node(
            "Gather", ["right", "pick_index"], ["right_b"], name="pick_right", axis=0
        ),
        helper.make_node("Sub", ["right_b", "left_b"], ["selected_width_minus_one"]),
        helper.make_node("Add", ["selected_width_minus_one", "one_i32"], ["Wb"]),
        helper.make_node(
            "Gather",
            ["rowpres", "pick_index"],
            ["selected_row_presence"],
            name="pick_row_presence",
            axis=0,
        ),
        helper.make_node(
            "Mul", ["selected_row_presence", "xidxf"], ["selected_row_forward"]
        ),
        helper.make_node(
            "ReduceMax",
            ["selected_row_forward"],
            ["selected_bottom_f16"],
            axes=[1],
            keepdims=0,
        ),
        helper.make_node(
            "Cast",
            ["selected_bottom_f16"],
            ["selected_bottom"],
            to=TensorProto.INT32,
        ),
        helper.make_node(
            "Mul", ["selected_row_presence", "xrevf"], ["selected_row_reverse"]
        ),
        helper.make_node(
            "ReduceMax",
            ["selected_row_reverse"],
            ["selected_top_reverse_f16"],
            axes=[1],
            keepdims=0,
        ),
        helper.make_node(
            "Cast",
            ["selected_top_reverse_f16"],
            ["selected_top_reverse"],
            to=TensorProto.INT32,
        ),
        helper.make_node(
            "Sub", ["nine_i32", "selected_top_reverse"], ["top_b"]
        ),
        helper.make_node(
            "Sub", ["selected_bottom", "top_b"], ["selected_height_minus_one"]
        ),
        helper.make_node("Add", ["selected_height_minus_one", "one_i32"], ["Hb"]),
    ]
    rewritten[pick_index + 1 : pick_index + 1] = replacement
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    used = {name for node in model.graph.node for name in node.input if name}
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    onnx.checker.check_model(onnx.load(output_path), full_check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute task174 vertical bbox only after selecting the symmetric object."
    )
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.parent, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
