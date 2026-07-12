from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    old = list(model.graph.node)
    if old[3].output != ["seed_color"] or old[9].output != ["seed_col_u8"]:
        raise RuntimeError("unexpected task141 graph")

    for name, value in {
        "flat_grid_shape": np.array([1, 900], dtype=np.int64),
        "thirty_i64": np.array(30, dtype=np.int64),
        "seed_unsqueeze_axes": np.array([1], dtype=np.int64),
    }.items():
        model.graph.initializer.append(numpy_helper.from_array(value, name=name))

    locator = [
        helper.make_node("Einsum", ["input", "color_idx"], ["scalar_grid"], equation="bchw,qcyx->bhw", name="decode_scalar_grid"),
        helper.make_node("ReduceMax", ["scalar_grid"], ["seed_color"], axes=[1, 2], keepdims=1, name="seed_color_from_grid"),
        helper.make_node("Reshape", ["scalar_grid", "flat_grid_shape"], ["flat_grid"], name="flatten_seed_grid"),
        helper.make_node("ArgMax", ["flat_grid"], ["seed_flat"], axis=1, keepdims=0, name="seed_flat_index"),
        helper.make_node("Div", ["seed_flat", "thirty_i64"], ["seed_row_i64"], name="seed_row_index"),
        helper.make_node("Mod", ["seed_flat", "thirty_i64"], ["seed_col_i64"], name="seed_col_index"),
        helper.make_node("Cast", ["seed_row_i64"], ["seed_row_vec"], to=TensorProto.UINT8, name="seed_row_to_u8"),
        helper.make_node("Cast", ["seed_col_i64"], ["seed_col_vec"], to=TensorProto.UINT8, name="seed_col_to_u8"),
        helper.make_node("Unsqueeze", ["seed_row_vec", "seed_unsqueeze_axes"], ["seed_row_u8"], name="seed_row_broadcast_shape"),
        helper.make_node("Unsqueeze", ["seed_col_vec", "seed_unsqueeze_axes"], ["seed_col_u8"], name="seed_col_broadcast_shape"),
    ]
    del model.graph.node[:]
    model.graph.node.extend(old[:3] + locator + old[10:])
    removed = {"coord_f"}
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
