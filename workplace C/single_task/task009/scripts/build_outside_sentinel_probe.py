from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def replace_initializer(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == name:
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(array, name=name))
            return
    model.graph.initializer.append(numpy_helper.from_array(array, name=name))


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))

    # Active cells remain encoded as their color (background=0, colors=1..9),
    # while all-zero padded input receives sentinel 10 from the Conv bias.
    conv = model.graph.node[0]
    if conv.op_type != "Conv" or list(conv.input) != ["input", "conv_w"]:
        raise RuntimeError("unexpected task009 source graph")
    conv.input.append("conv_b")
    replace_initializer(
        model,
        "conv_w",
        np.arange(-10.0, 0.0, dtype=np.float32).reshape(1, 10, 1, 1),
    )
    replace_initializer(model, "conv_b", np.array([10.0], dtype=np.float32))
    replace_initializer(model, "outside_u8", np.array(10, dtype=np.uint8))

    rewritten = []
    for node in model.graph.node:
        if node.output and node.output[0] == "valid_b":
            node.op_type = "Equal"
            del node.attribute[:]
            del node.input[:]
            node.input.extend(["cgrid_u8", "outside_u8"])
            node.output[0] = "outside_b"
            rewritten.append(node)
            continue
        if node.output and node.output[0] == "content_grid":
            # content_u8 already contains sentinel 10 outside the valid grid.
            continue
        if node.output and node.output[0] == "line_grid":
            del node.input[:]
            node.input.extend(["outside_b", "outside_u8", "line_u8"])
        for index, value in enumerate(node.input):
            if value == "content_grid":
                node.input[index] = "content_u8"
            elif value == "u255":
                node.input[index] = "outside_u8"
        rewritten.append(node)

    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    kept_initializers = [item for item in model.graph.initializer if item.name != "u255"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)

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
