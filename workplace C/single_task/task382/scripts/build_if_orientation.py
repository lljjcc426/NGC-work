from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    replacement = []
    inserted = False

    for node in model.graph.node:
        if node.op_type == "Transpose" and list(node.output) == ["mask_t"]:
            continue
        if node.op_type == "Where" and list(node.output) == ["oriented_u"]:
            then_graph = helper.make_graph(
                [helper.make_node("Identity", ["mask_canon"], ["then_mask"])],
                "orientation_identity",
                [],
                [helper.make_tensor_value_info("then_mask", TensorProto.UINT8, [20, 20])],
            )
            else_graph = helper.make_graph(
                [helper.make_node("Transpose", ["mask_canon"], ["else_mask"], perm=[1, 0])],
                "orientation_transpose",
                [],
                [helper.make_tensor_value_info("else_mask", TensorProto.UINT8, [20, 20])],
            )
            replacement.append(
                helper.make_node(
                    "If",
                    ["is_h"],
                    ["oriented_u"],
                    name="select_orientation",
                    then_branch=then_graph,
                    else_branch=else_graph,
                )
            )
            inserted = True
            continue
        replacement.append(deepcopy(node))

    if not inserted:
        raise RuntimeError("task382 orientation nodes not found")

    del model.graph.node[:]
    model.graph.node.extend(replacement)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.source, args.output))


if __name__ == "__main__":
    main()
