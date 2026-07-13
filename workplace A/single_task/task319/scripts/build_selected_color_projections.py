from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper, shape_inference


DEFAULT_SOURCE = Path(
    r"E:/kagglegolf/submissions/candidates/"
    r"GOLF_20260713_SUBMISSION8_REBASE/onnx/task319.onnx"
)
DEFAULT_OUTPUT = Path(__file__).parents[1] / "onnx" / "task319_selected_color_projections.onnx"


def build(source: Path, output: Path) -> Path:
    model = onnx.load(source)
    nodes = list(model.graph.node)

    expected = {
        0: ("Einsum", "safe_name_24"),
        1: ("Einsum", "safe_name_25"),
        2: ("ReduceSum", "safe_name_26"),
        16: ("Gather", "safe_name_43"),
        17: ("Gather", "safe_name_44"),
        18: ("Gather", "safe_name_45"),
        19: ("Gather", "safe_name_46"),
        20: ("Gather", "safe_name_47"),
        21: ("Gather", "safe_name_48"),
    }
    for index, (op_type, output_name) in expected.items():
        node = nodes[index]
        if node.op_type != op_type or list(node.output) != [output_name]:
            raise ValueError(
                f"unexpected parent node {index}: {node.op_type} {list(node.output)}"
            )

    rewritten: list[onnx.NodeProto] = []
    for index, node in enumerate(nodes):
        if index == 0:
            # Select the three non-background colors from total cell counts without
            # first materializing per-column counts for all ten colors.
            rewritten.append(
                helper.make_node(
                    "Einsum",
                    ["input"],
                    ["safe_name_26"],
                    name="task319_direct_color_counts",
                    equation="bchw->c",
                )
            )
            continue
        if index in {1, 2}:
            continue
        if index == 16:
            rewritten.extend(
                [
                    helper.make_node(
                        "Equal",
                        ["safe_name_2", "safe_name_39"],
                        ["task319_color3_selector_bool"],
                        name="task319_color3_selector",
                    ),
                    helper.make_node(
                        "Cast",
                        ["safe_name_34"],
                        ["task319_color1_selector"],
                        name="task319_color1_selector_cast",
                        to=TensorProto.FLOAT,
                    ),
                    helper.make_node(
                        "Cast",
                        ["safe_name_37"],
                        ["task319_color2_selector"],
                        name="task319_color2_selector_cast",
                        to=TensorProto.FLOAT,
                    ),
                    helper.make_node(
                        "Cast",
                        ["task319_color3_selector_bool"],
                        ["task319_color3_selector"],
                        name="task319_color3_selector_cast",
                        to=TensorProto.FLOAT,
                    ),
                ]
            )
            for rank, (selector, row_output, column_output) in enumerate(
                [
                    ("task319_color1_selector", "safe_name_43", "safe_name_44"),
                    ("task319_color2_selector", "safe_name_45", "safe_name_46"),
                    ("task319_color3_selector", "safe_name_47", "safe_name_48"),
                ],
                start=1,
            ):
                rewritten.extend(
                    [
                        helper.make_node(
                            "Einsum",
                            ["input", selector, "safe_name_0"],
                            [row_output],
                            name=f"task319_selected_row_bits_{rank}",
                            equation="bchw,c,w->h",
                        ),
                        helper.make_node(
                            "Einsum",
                            ["input", selector],
                            [column_output],
                            name=f"task319_selected_column_counts_{rank}",
                            equation="bchw,c->w",
                        ),
                    ]
                )
            continue
        if 17 <= index <= 21:
            continue
        rewritten.append(node)

    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    stale = {"safe_name_24", "safe_name_25"}
    kept_value_info = [v for v in model.graph.value_info if v.name not in stale]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)

    onnx.checker.check_model(model, full_check=True)
    model = shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build(args.source, args.output))


if __name__ == "__main__":
    main()
