from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _replace_initializer(
    model: onnx.ModelProto,
    name: str,
    value: np.ndarray,
) -> None:
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == name:
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(value, name=name)
            )
            return
    raise RuntimeError(f"initializer not found: {name}")


def build(source: Path, output: Path, keep: int = 10) -> Path:
    model = deepcopy(onnx.load(source))
    nodes = {
        value: node
        for node in model.graph.node
        for value in node.output
        if value
    }
    required = {
        "safe_name_67",
        "safe_name_75",
        "safe_name_76",
        "safe_name_78",
        "safe_name_80",
        "safe_name_82",
    }
    if not required.issubset(nodes):
        raise RuntimeError(f"unexpected task035 graph: {sorted(required - set(nodes))}")

    colors_i32 = "e35_colors_i32"
    selected_colors_i32 = "e35_selected_colors_i32"
    selected_colors = "e35_selected_colors"
    selected_slots = "e35_selected_slots"
    selected_rows = "e35_selected_rows"
    selected_cols = "e35_selected_cols"
    selection_nodes = [
        helper.make_node(
            "Cast",
            ["safe_name_67"],
            [colors_i32],
            to=onnx.TensorProto.INT32,
            name="e35_colors_to_i32",
        ),
        helper.make_node(
            "TopK",
            [colors_i32, "e35_keep"],
            [selected_colors_i32, selected_slots],
            axis=1,
            largest=1,
            sorted=0,
            name="e35_select_nonzero_slots",
        ),
        helper.make_node(
            "Cast",
            [selected_colors_i32],
            [selected_colors],
            to=onnx.TensorProto.UINT8,
            name="e35_selected_colors_to_u8",
        ),
        helper.make_node(
            "GatherElements",
            ["safe_name_75", selected_slots],
            [selected_rows],
            axis=1,
            name="e35_select_rows",
        ),
        helper.make_node(
            "GatherElements",
            ["safe_name_76", selected_slots],
            [selected_cols],
            axis=1,
            name="e35_select_cols",
        ),
    ]

    nodes["safe_name_78"].input[0] = selected_colors
    nodes["safe_name_80"].input[0] = selected_rows
    nodes["safe_name_82"].input[0] = selected_cols

    rebuilt: list[onnx.NodeProto] = []
    inserted = False
    for node in model.graph.node:
        if not inserted and node is nodes["safe_name_78"]:
            rebuilt.extend(selection_nodes)
            inserted = True
        rebuilt.append(deepcopy(node))
    if not inserted:
        raise RuntimeError("task035 scatter assembly insertion point not found")
    del model.graph.node[:]
    model.graph.node.extend(rebuilt)

    _replace_initializer(
        model,
        "safe_name_31",
        np.zeros((1, keep * 2, 1), dtype=np.uint8),
    )
    _replace_initializer(
        model,
        "safe_name_32",
        np.concatenate(
            [
                np.zeros((1, keep), dtype=np.float32),
                np.ones((1, keep), dtype=np.float32),
            ],
            axis=1,
        ),
    )
    _replace_initializer(
        model,
        "concat_pad_1",
        np.asarray([keep, 0], dtype=np.int64),
    )
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray([keep], dtype=np.int64), name="e35_keep")
    )

    model.producer_name = "ngc_e_task035_compact_scatter"
    onnx.checker.check_model(model, full_check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep", type=int, default=10)
    args = parser.parse_args()
    print(build(args.source, args.output, keep=args.keep))


if __name__ == "__main__":
    main()
