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
    if old[24].op_type != "Gather" or old[25].op_type != "Gather":
        raise RuntimeError("unexpected task388 graph")

    for name, value in {
        "axis0_u": np.array([0], dtype=np.int64),
        "tile_starts": np.array([0, 0], dtype=np.int64),
        "tile_repeats": np.array([1, 1, 2, 2], dtype=np.int64),
        "thirty_i32": np.array(30, dtype=np.int32),
        "pad_begin_hw": np.array([0, 0], dtype=np.int64),
    }.items():
        model.graph.initializer.append(numpy_helper.from_array(value, name=name))

    tiling = [
        helper.make_node("Cast", ["N"], ["N_i64"], to=TensorProto.INT64, name="N_to_i64"),
        helper.make_node("Concat", ["N_i64", "N_i64"], ["tile_ends"], axis=0, name="tile_crop_ends"),
        helper.make_node("Slice", ["base7", "tile_starts", "tile_ends", "axes_hw"], ["base_N"], name="crop_base_to_N"),
        helper.make_node("Tile", ["base_N", "tile_repeats"], ["target_2N"], name="tile_two_by_two"),
        helper.make_node("Sub", ["thirty_i32", "twoN"], ["pad_end_i32"], name="remaining_output_pad"),
        helper.make_node("Cast", ["pad_end_i32"], ["pad_end_i64"], to=TensorProto.INT64, name="pad_end_to_i64"),
        helper.make_node("Concat", ["pad_begin_hw", "pad_end_i64", "pad_end_i64"], ["dynamic_pads"], axis=0, name="dynamic_hw_pads"),
        helper.make_node("Pad", ["target_2N", "dynamic_pads", "u255", "axes_hw"], ["idx30"], mode="constant", name="pad_tiled_output"),
        old[27],
    ]
    del model.graph.node[:]
    model.graph.node.extend(old[:19] + tiling)

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
