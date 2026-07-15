from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


REMOVED_OUTPUTS = {
    "selected_row_forward",
    "selected_bottom_f16",
    "selected_bottom",
    "selected_row_reverse",
    "selected_top_reverse_f16",
    "selected_top_reverse",
    "top_b",
}


def build(parent_path: Path, output_path: Path) -> None:
    model = onnx.load(parent_path)
    nodes = list(model.graph.node)
    removed = {
        index
        for index, node in enumerate(nodes)
        if any(output in REMOVED_OUTPUTS for output in node.output)
    }
    if len(removed) != len(REMOVED_OUTPUTS):
        raise RuntimeError("unexpected task174 late-bbox chain")
    rewritten = [node for index, node in enumerate(nodes) if index not in removed]
    insertion = next(
        index
        for index, node in enumerate(rewritten)
        if "selected_row_presence" in node.output
    ) + 1
    replacement = [
        helper.make_node(
            "ArgMax",
            ["selected_row_presence"],
            ["selected_top_i64"],
            name="selected_first_row",
            axis=1,
            keepdims=0,
            select_last_index=0,
        ),
        helper.make_node(
            "Cast",
            ["selected_top_i64"],
            ["top_b"],
            name="selected_top_i32",
            to=TensorProto.INT32,
        ),
        helper.make_node(
            "ArgMax",
            ["selected_row_presence"],
            ["selected_bottom_i64"],
            name="selected_last_row",
            axis=1,
            keepdims=0,
            select_last_index=1,
        ),
        helper.make_node(
            "Cast",
            ["selected_bottom_i64"],
            ["selected_bottom"],
            name="selected_bottom_i32",
            to=TensorProto.INT32,
        ),
    ]
    rewritten[insertion:insertion] = replacement
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
        description="Use first/last ArgMax for task174 selected-object vertical bounds."
    )
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.parent, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
