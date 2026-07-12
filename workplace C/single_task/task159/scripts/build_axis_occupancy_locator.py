from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    names = {node.output[0]: index for index, node in enumerate(model.graph.node) if node.output}
    if "sprite_row_code" not in names or "left_i32" not in names:
        raise RuntimeError("unexpected task159 graph")

    locator = [
        helper.make_node("Einsum", ["input", "motif_mask"], ["row_counts"], equation="bcij,c->i", name="motif_row_counts"),
        helper.make_node("Sign", ["row_counts"], ["row_present"], name="motif_rows_present"),
        helper.make_node("ArgMax", ["row_present"], ["top_i64"], axis=0, keepdims=1, name="first_motif_row"),
        helper.make_node("Cast", ["top_i64"], ["top_i32"], to=TensorProto.INT32, name="top_to_i32"),
        helper.make_node("Einsum", ["input", "motif_mask"], ["col_counts"], equation="bcij,c->j", name="motif_col_counts"),
        helper.make_node("Sign", ["col_counts"], ["col_present"], name="motif_cols_present"),
        helper.make_node("ArgMax", ["col_present"], ["left_i64"], axis=0, keepdims=1, name="first_motif_col"),
        helper.make_node("Cast", ["left_i64"], ["left_i32"], to=TensorProto.INT32, name="left_to_i32"),
    ]
    old = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(old[:6] + locator + old[16:])
    removed = {"desc30", "inv_ln4", "const_29_vec"}
    kept = [item for item in model.graph.initializer if item.name not in removed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
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
