from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    old = list(model.graph.node)
    if old[0].op_type != "Conv" or old[0].output != ["color_grid_f_full"]:
        raise RuntimeError("unexpected task046 graph")
    for name, value in {
        "decode_starts": np.array([0, 0], dtype=np.int64),
        "decode_ends": np.array([3, 20], dtype=np.int64),
        "decode_axes": np.array([2, 3], dtype=np.int64),
    }.items():
        model.graph.initializer.append(numpy_helper.from_array(value, name=name))
    decode = [
        helper.make_node("Slice", ["input", "decode_starts", "decode_ends", "decode_axes"], ["decode_patch"], name="crop_path_area"),
        helper.make_node("Einsum", ["decode_patch", "color_weights"], ["color_grid_f_full"], equation="bchw,ockl->bohw", name="decode_path_colors"),
    ]
    del model.graph.node[:]
    model.graph.node.extend(decode + old[1:])
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
