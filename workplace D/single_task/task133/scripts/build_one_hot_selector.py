from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


EXPECTED_OUTPUTS = {
    "pcds",
    "Pcode",
    "Pcode_u8",
    "ptop",
    "Pt",
    "pleft",
    "Pl",
}


def build(parent_path: Path, output_path: Path) -> None:
    model = onnx.load(parent_path)
    nodes = list(model.graph.node)
    output_to_index = {
        output: index
        for index, node in enumerate(nodes)
        for output in node.output
    }
    indices = sorted(output_to_index[name] for name in EXPECTED_OUTPUTS)
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise RuntimeError("task133 selector chain is not contiguous")
    if [nodes[index].op_type for index in indices] != [
        "Mul",
        "ReduceSum",
        "Cast",
        "Mul",
        "ReduceSum",
        "Mul",
        "ReduceSum",
    ]:
        raise RuntimeError("unexpected task133 selector chain")
    consumers = {
        name: [node for node in nodes for value in node.input if value == name]
        for name in EXPECTED_OUTPUTS
    }
    for intermediate in ("pcds", "Pcode", "ptop", "pleft"):
        if len(consumers[intermediate]) != 1:
            raise RuntimeError(f"unexpected consumers for {intermediate}")

    replacement = [
        helper.make_node(
            "ArgMax",
            ["pbm"],
            ["Pcode_i64"],
            name="select_code_once",
            axis=1,
            keepdims=1,
        ),
        helper.make_node(
            "Add",
            ["Pcode_i64", "selector_one_i64"],
            ["Pcode_one_based"],
            name="selected_code_offset",
        ),
        helper.make_node(
            "Cast",
            ["Pcode_one_based"],
            ["Pcode_u8"],
            name="selected_code_u8",
            to=TensorProto.UINT8,
        ),
        helper.make_node(
            "GatherElements",
            ["topf", "Pcode_i64"],
            ["Pt"],
            name="selected_top",
            axis=1,
        ),
        helper.make_node(
            "GatherElements",
            ["leftf", "Pcode_i64"],
            ["Pl"],
            name="selected_left",
            axis=1,
        ),
    ]
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray(1, dtype=np.int64), "selector_one_i64")
    )
    start = indices[0]
    del model.graph.node[indices[0] : indices[-1] + 1]
    for offset, node in enumerate(replacement):
        model.graph.node.insert(start + offset, node)

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
        description="Replace task133 one-hot mask reductions with scalar selection."
    )
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.parent, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
