from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


def replace_initializer(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == name:
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(array, name=name))
            return
    raise KeyError(name)


def build(source: Path, output: Path, remove_redundant_casts: bool) -> Path:
    model = deepcopy(onnx.load(source))

    for node in model.graph.node:
        if node.output and node.output[0] == "onehot16":
            if node.op_type != "Cast":
                raise RuntimeError("onehot16 is not produced by Cast")
            for attribute in node.attribute:
                if attribute.name == "to":
                    attribute.i = TensorProto.UINT8
                    break
            node.output[0] = "onehot_u8"
        for index, value in enumerate(node.input):
            if value == "onehot16":
                node.input[index] = "onehot_u8"

    replace_initializer(model, "cv16", numpy_helper.to_array(next(x for x in model.graph.initializer if x.name == "cv16")).astype(np.uint8))
    replace_initializer(model, "sepA", numpy_helper.to_array(next(x for x in model.graph.initializer if x.name == "sepA")).astype(np.uint8))

    if remove_redundant_casts:
        rewrites = {"h_u8": "h_between", "v_u8": "v_between"}
        kept = []
        for node in model.graph.node:
            if node.op_type == "Cast" and node.output and node.output[0] in rewrites:
                continue
            for index, value in enumerate(node.input):
                if value in rewrites:
                    node.input[index] = rewrites[value]
            kept.append(node)
        del model.graph.node[:]
        model.graph.node.extend(kept)

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-casts", action="store_true")
    args = parser.parse_args()
    print(build(args.source, args.output, remove_redundant_casts=not args.keep_casts))


if __name__ == "__main__":
    main()
