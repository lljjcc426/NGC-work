from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


REMOVED_OUTPUTS = {"cx", "ru", "right", "cxr", "lu", "lrev", "left"}


def build(parent_path: Path, output_path: Path) -> None:
    model = onnx.load(parent_path)
    nodes = list(model.graph.node)
    removed = {
        index
        for index, node in enumerate(nodes)
        if any(output in REMOVED_OUTPUTS for output in node.output)
    }
    if len(removed) != len(REMOVED_OUTPUTS):
        raise RuntimeError("unexpected task174 horizontal-bound chain")
    rewritten = [node for index, node in enumerate(nodes) if index not in removed]
    insertion = next(
        index for index, node in enumerate(rewritten) if "colpres" in node.output
    ) + 1
    replacement = [
        helper.make_node(
            "ArgMax",
            ["colpres"],
            ["left_i64"],
            name="first_object_column",
            axis=2,
            keepdims=0,
            select_last_index=0,
        ),
        helper.make_node(
            "Cast", ["left_i64"], ["left"], name="left_i32", to=TensorProto.INT32
        ),
        helper.make_node(
            "ArgMax",
            ["colpres"],
            ["right_i64"],
            name="last_object_column",
            axis=2,
            keepdims=0,
            select_last_index=1,
        ),
        helper.make_node(
            "Cast",
            ["right_i64"],
            ["right"],
            name="right_i32",
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
        description="Use first/last ArgMax for task174 candidate-object horizontal bounds."
    )
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.parent, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
