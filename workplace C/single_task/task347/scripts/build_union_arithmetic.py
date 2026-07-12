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
    if [node.op_type for node in old] != ["Slice", "Slice", "Greater", "Where", "MaxUnpool"]:
        raise RuntimeError("unexpected task347 graph")

    # Decode the explicit union rule: color 4 in the left panel OR color 3 in
    # the right panel. The two-channel tensor contains background and color 6.
    right3 = deepcopy(old[1])
    right3.name = "slice_right_color3"
    right3.output[0] = "right3"
    for attribute in right3.attribute:
        if attribute.name == "starts":
            attribute.ints[:] = [3, 0, 3]
        elif attribute.name == "ends":
            attribute.ints[:] = [4, 3, 6]

    model.graph.initializer.append(numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="one_f"))
    arithmetic = [
        old[0],
        right3,
        helper.make_node("Max", ["left4", "right3"], ["union_f"], name="panel_union"),
        helper.make_node("Sub", ["one_f", "union_f"], ["blank_f"], name="panel_background"),
        helper.make_node("Concat", ["blank_f", "union_f"], ["small2_f"], axis=1, name="two_color_output"),
        helper.make_node("Cast", ["small2_f"], ["small2"], to=TensorProto.FLOAT16, name="small_to_float16"),
        old[4],
    ]
    del model.graph.node[:]
    model.graph.node.extend(arithmetic)
    removed = {"blank_template", "union_template"}
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
