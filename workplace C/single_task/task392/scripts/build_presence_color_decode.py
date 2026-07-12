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
    if old[0].output != ["num"] or old[3].output != ["color_u8"]:
        raise RuntimeError("unexpected task392 graph")
    model.graph.initializer.append(
        numpy_helper.from_array(np.array([0, 2, 3], dtype=np.int64), name="presence_axes")
    )
    decode = [
        helper.make_node("ReduceMax", ["input", "presence_axes"], ["channel_present"], keepdims=0, name="channel_presence"),
        helper.make_node("Mul", ["channel_present", "chanmask"], ["nonbg_present"], name="exclude_background"),
        helper.make_node("ArgMax", ["nonbg_present"], ["color_i64"], axis=0, keepdims=0, name="present_color"),
        helper.make_node("Cast", ["color_i64"], ["color_u8"], to=TensorProto.UINT8, name="color_to_u8"),
    ]
    del model.graph.node[:]
    model.graph.node.extend(decode + old[4:])
    kept = [item for item in model.graph.initializer if item.name != "arange_color"]
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
