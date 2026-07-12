from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    first = model.graph.node[0]
    if first.op_type != "Einsum" or first.output != ["templ_scalar_f"]:
        raise RuntimeError("unexpected task075 source graph")

    added = {
        "template_starts": np.array([0, 0], dtype=np.int64),
        "template_ends": np.array([3, 3], dtype=np.int64),
        "template_axes": np.array([2, 3], dtype=np.int64),
    }
    for name, value in added.items():
        model.graph.initializer.append(numpy_helper.from_array(value, name=name))

    replacement = [
        helper.make_node(
            "Slice",
            ["input", "template_starts", "template_ends", "template_axes"],
            ["template_onehot"],
            name="slice_top_left_template",
        ),
        helper.make_node(
            "ArgMax",
            ["template_onehot"],
            ["templ_scalar_f"],
            name="decode_template_color",
            axis=1,
            keepdims=1,
        ),
    ]
    old = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(replacement + old[1:])
    removed = {"color_weights", "sel3", "one_k"}
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
