from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


INT8_INITIALIZERS = {"zero_f16", "c1_f16", "coords20"}
INT8_CAST_OUTPUTS = {
    "contig2_f16",
    "contig3_f16",
    "contig4_f16",
    "contig5_f16",
    "rows_col",
    "cols_col",
    "step_c_f16",
    "step_r_f16",
}


def rewrite(source: Path, destination: Path) -> None:
    model = onnx.version_converter.convert_version(onnx.load(source), 14)

    converted_constants = {
        node.output[0]
        for node in model.graph.node
        if node.op_type == "Constant" and node.output
    }
    unsqueezes = 0
    splits = 0
    for node in model.graph.node:
        if node.op_type == "Unsqueeze" and len(node.input) == 2:
            node.input[1] = "task280_axes1"
            unsqueezes += 1
        elif node.op_type == "Split" and len(node.input) == 2:
            node.input[1] = "task280_split11"
            splits += 1
    kept_nodes = [
        node
        for node in model.graph.node
        if not (node.op_type == "Constant" and node.output[0] in converted_constants)
    ]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(
                np.array([1], dtype=np.int64), name="task280_axes1"
            ),
            numpy_helper.from_array(
                np.array([1, 1], dtype=np.int64), name="task280_split11"
            ),
        ]
    )
    if len(converted_constants) != 15 or unsqueezes != 9 or splits != 6:
        raise RuntimeError(
            "unexpected opset conversion: "
            f"constants={len(converted_constants)}, unsqueeze={unsqueezes}, split={splits}"
        )

    converted = 0
    for index, item in enumerate(model.graph.initializer):
        if item.name not in INT8_INITIALIZERS:
            continue
        array = numpy_helper.to_array(item).astype(np.int8)
        model.graph.initializer[index].CopyFrom(
            numpy_helper.from_array(array, name=item.name)
        )
        converted += 1

    narrowed_casts = 0
    for node in model.graph.node:
        if node.op_type != "Cast" or not node.output:
            continue
        if node.output[0] not in INT8_CAST_OUTPUTS:
            continue
        for attr in node.attribute:
            if attr.name == "to":
                attr.i = TensorProto.INT8
                narrowed_casts += 1
                break

    if converted != len(INT8_INITIALIZERS) or narrowed_casts != len(INT8_CAST_OUTPUTS):
        raise RuntimeError(
            f"unexpected graph: initializers={converted}, casts={narrowed_casts}"
        )

    rewritten: list[onnx.NodeProto] = []
    replaced_sum = False
    for node in model.graph.node:
        if node.op_type == "Sum" and list(node.output) == ["inward_count_col5"]:
            inputs = list(node.input)
            if inputs != [
                "c1_f16",
                "contig2_f16",
                "contig3_f16",
                "contig4_f16",
                "contig5_f16",
            ]:
                raise RuntimeError(f"unexpected thickness inputs: {inputs}")
            rewritten.extend(
                [
                    helper.make_node("Add", inputs[:2], ["task280_count2_i8"]),
                    helper.make_node(
                        "Add",
                        ["task280_count2_i8", inputs[2]],
                        ["task280_count3_i8"],
                    ),
                    helper.make_node(
                        "Add",
                        ["task280_count3_i8", inputs[3]],
                        ["task280_count4_i8"],
                    ),
                    helper.make_node(
                        "Add",
                        ["task280_count4_i8", inputs[4]],
                        ["inward_count_col5"],
                    ),
                ]
            )
            replaced_sum = True
        else:
            rewritten.append(node)
    if not replaced_sum:
        raise RuntimeError("thickness Sum node not found")
    del model.graph.node[:]
    model.graph.node.extend(rewritten)

    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    print(
        f"saved={destination} nodes={len(model.graph.node)} "
        f"int8_initializers={converted} int8_casts={narrowed_casts}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    rewrite(args.source, args.destination)


if __name__ == "__main__":
    main()
