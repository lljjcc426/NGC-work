from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(parent_path: Path, output_path: Path) -> None:
    model = onnx.load(parent_path)
    nodes = list(model.graph.node)
    start = next(index for index, node in enumerate(nodes) if node.output and node.output[0] == "pick")
    end = next(index for index, node in enumerate(nodes) if node.output and node.output[0] == "kcol32")
    expected = [
        "Cast", "Cast", "Mul", "ReduceMax", "Mul", "ReduceMax", "Mul", "ReduceMax",
        "Mul", "ReduceMax", "Cast", "Where", "ReduceMax", "Cast", "Mul", "ReduceMax",
    ]
    if [node.op_type for node in nodes[start : end + 1]] != expected:
        raise RuntimeError("unexpected task174 selector chain")

    axes_name = "scalar_pick_squeeze_axis"
    model.graph.initializer.append(numpy_helper.from_array(np.asarray([1], dtype=np.int64), axes_name))
    replacement = [
        helper.make_node("Squeeze", ["e174_hash_all_u8", axes_name], ["pick3"], name="pick_to_vector"),
        helper.make_node("ArgMax", ["pick3"], ["pick_index"], name="scalar_pick", axis=0, keepdims=0),
        helper.make_node("Gather", ["top", "pick_index"], ["top_b"], name="pick_top", axis=0),
        helper.make_node("Gather", ["left", "pick_index"], ["left_b"], name="pick_left", axis=0),
        helper.make_node("Gather", ["H_i", "pick_index"], ["Hb"], name="pick_height", axis=0),
        helper.make_node("Gather", ["W_i", "pick_index"], ["Wb"], name="pick_width", axis=0),
        helper.make_node("Gather", ["col_f", "pick_index"], ["kcol32"], name="pick_color", axis=0),
        helper.make_node("Cast", ["kcol32"], ["kcolf"], name="pick_color_f16", to=TensorProto.FLOAT16),
    ]
    del model.graph.node[start : end + 1]
    for offset, node in enumerate(replacement):
        model.graph.node.insert(start + offset, node)
    used = {name for node in model.graph.node for name in node.input if name}
    kept = [initializer for initializer in model.graph.initializer if initializer.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    onnx.checker.check_model(onnx.load(output_path), full_check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Select task174's unique symmetric object once and gather its scalars.")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.parent, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
